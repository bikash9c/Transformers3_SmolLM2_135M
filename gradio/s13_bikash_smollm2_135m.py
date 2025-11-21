# ---- SmolLM2-135M Model for Inference ----

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------- RMSNorm ----------------
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

# ---------------- Rotary Embeddings ----------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len: Optional[int] = None):
        if seq_len > self.cos_cached.size(0):
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# ---------------- Attention (GQA) ----------------
class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _repeat_kv(self, x, n_rep):
        b, n_kv, s, d = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)

    def forward(self, x, cos=None, sin=None):
        b, s, _ = x.size()

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, s, self.hidden_size)
        return self.o_proj(attn)

# ---------------- MLP ----------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ---------------- Decoder Block ----------------
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = MLP(config)
        self.input_norm = RMSNorm(config.hidden_size)
        self.post_norm = RMSNorm(config.hidden_size)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_norm(x), cos, sin)
        x = x + self.mlp(self.post_norm(x))
        return x

# ---------------- Config ----------------
@dataclass
class SmolLM2Config:
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True

# ---------------- Model ----------------
class SmolLM2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.rotary = RotaryEmbedding(config.hidden_size // config.num_attention_heads,
                                      max_position_embeddings=config.max_position_embeddings,
                                      base=config.rope_theta)
        self.config = config

    def forward(self, input_ids):
        b, s = input_ids.size()
        x = self.embed(input_ids)
        cos, sin = self.rotary(x, seq_len=s)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

class SmolLM2ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = SmolLM2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed.weight
        self.config = config

    def forward(self, input_ids):
        hidden = self.model(input_ids)
        return None, self.lm_head(hidden)

    # ----- generation -----
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            with torch.no_grad():
                _, logits = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                vals = torch.topk(logits, top_k)[0][..., -1, None]
                logits[logits < vals] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == self.config.eos_token_id:
                break
        return input_ids
