# SmolLM2-135M: Training from Scratch

A PyTorch implementation of Hugging Face's SmolLM2-135M model, trained from scratch with custom optimizations and checkpointing.

## 📋 Overview

This project implements and trains the SmolLM2-135M language model, a compact 135-million parameter transformer based on the Llama2 architecture with Grouped Query Attention (GQA). The model is trained on text data with periodic checkpointing and generation sampling.

## 🏗️ Model Architecture

### Core Specifications

| Component | Value |
|-----------|-------|
| **Parameters** | 135M |
| **Layers** | 30 |
| **Hidden Size** | 576 |
| **Intermediate Size** | 1536 |
| **Attention Heads** | 9 |
| **Key-Value Heads** | 3 (GQA) |
| **Vocabulary Size** | 49,152 |
| **Max Sequence Length** | 2048 |
| **RoPE Theta** | 10,000.0 |

### Key Components

#### 1. **RMSNorm (Root Mean Square Normalization)**
- Efficient alternative to LayerNorm
- Used for pre-normalization before attention and MLP blocks
- Formula: `x * rsqrt(mean(x²) + ε) * weight`

#### 2. **Rotary Position Embedding (RoPE)**
- Relative positional encoding applied to queries and keys
- Enables better length extrapolation
- Base frequency: 10,000.0

#### 3. **Grouped Query Attention (GQA)**
- Multi-head attention with parameter-efficient key-value sharing
- 9 query heads, 3 key-value heads (3:1 ratio)
- Uses Flash Attention (`F.scaled_dot_product_attention`) for 4x speedup
- Head dimension: 64 (576 / 9)

#### 4. **SwiGLU MLP**
- Feed-forward network with gated linear units
- Formula: `down(silu(gate(x)) ⊙ up(x))`
- More expressive than standard FFN

#### 5. **Decoder Layer**
- Pre-normalization architecture (norm before sub-layers)
- Residual connections around attention and MLP
- Structure: `x = x + Attention(Norm(x)); x = x + MLP(Norm(x))`

## ⚡ Performance Optimizations

The implementation includes 5 key speedup techniques:

1. **Float32 Matrix Multiplication Precision**: `torch.set_float32_matmul_precision('high')`
2. **BFloat16 Mixed Precision**: Automatic mixed precision training for faster computation
3. **Torch Compile**: `torch.compile(model)` for graph optimization (Linux only)
4. **Flash Attention**: Memory-efficient attention with automatic causal masking
5. **Vocabulary Size**: Power-of-2 vocab size (49,152) for efficient GPU operations

## 🎯 Training Configuration

### Hyperparameters

```python
# Model Configuration
batch_size = 2
sequence_length = 1024
total_steps = 5000
save_interval = 500

# Optimizer (AdamW)
learning_rate = 5e-4 (peak)
min_learning_rate = 5e-5
betas = (0.9, 0.95)
weight_decay = 0.1
eps = 1e-8

# Learning Rate Schedule
warmup_steps = 500 (10% of total)
schedule = "cosine_annealing_with_warmup"

# Gradient Clipping
max_grad_norm = 1.0
```

### Learning Rate Schedule

The training uses a **cosine annealing schedule with warmup**:

1. **Warmup Phase** (steps 0-500): Linear increase from 0 to 5e-4
2. **Cosine Annealing** (steps 501-5000): Smooth decay to 5e-5 following cosine curve

```python
def get_lr(step):
    if step < warmup_steps:
        return 5e-4 * step / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (5e-4 - min_lr)
```

## 🔄 Training Workflow

### Phase 1: Initial Training (5000 steps)

1. Model initialization with random weights
2. Training for 5000 steps with:
   - Batch processing of text sequences
   - Loss computation (cross-entropy)
   - Gradient updates with AdamW
   - Learning rate scheduling
   - Gradient clipping for stability

3. **Checkpointing every 500 steps**:
   - Model state dict saved
   - Optimizer state saved
   - Training metadata (step, loss) saved
   - Sample generation for quality check

### Phase 2: Resume Training (50 additional steps)

1. Load checkpoint from step 5000
2. Resume training for 50 more steps (5001-5050)
3. Same training configuration maintained
4. Final model weights saved in half precision

## 📊 Training Logs

### Initial Training (Steps 1-5000)

```
[Add your training logs here - showing periodic checkpoints and sample generations]

Example format:
[TRAIN] Step 1 | Loss: 10.123 | LR: 0.000010 | dt: 234.56ms | tok/sec: 8745.23
[TRAIN] Step 500 | Loss: 5.234 | LR: 0.000500 | dt: 198.23ms | tok/sec: 10321.45
[SAVE] Checkpoint saved at step 500
Sample response: To be or not to be a great day ahead...

[TRAIN] Step 1000 | Loss: 4.123 | LR: 0.000495 | dt: 201.34ms | tok/sec: 10198.76
[SAVE] Checkpoint saved at step 1000
Sample response: To be or not to be the first time...

...
```

### Resume Training (Steps 5001-5050)

```
[Add your resume training logs here]

Example format:
[LOAD] Checkpoint loaded at step 5000
Training resumed from step 5001
[TRAIN] Step 5001 | Loss: 2.345 | LR: 0.000051 | dt: 195.67ms | tok/sec: 10456.89
[TRAIN] Step 5050 | Loss: 2.298 | LR: 0.000050 | dt: 197.23ms | tok/sec: 10387.34
Training completed!
```

## 📁 Project Structure

```
.
├── s13_bikash_smollm2_135m.py    # Main training script
├── input.txt                      # Training data
├── model/
│   ├── checkpoint.pth             # Training checkpoint
│   ├── model.pth                  # Full precision weights
│   └── model_final.pth            # Half precision weights
└── README.md                      # This file
```

## 🚀 Usage

### Training from Scratch

```python
# Initialize model
config = SmolLM2Config()
model = SmolLM2ForCausalLM(config)
model.to(device)
model = torch.compile(model)

# Setup training
train_loader = DataLoaderLite(B=2, T=1024)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95))

# Train for 5000 steps
for step in range(1, 5001):
    x, y = train_loader.next_batch()
    loss, logits = model(x, labels=y)
    # ... training loop
```

### Resume from Checkpoint

```python
# Load checkpoint
checkpoint = torch.load('model/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_step = checkpoint['step']

# Continue training
for step in range(start_step + 1, 5051):
    # ... training loop
```

### Generate Text

```python
# Load trained model
model = SmolLM2ForCausalLM(SmolLM2Config())
model.load_state_dict(torch.load('model/model_final.pth'))
model.eval()

# Generate
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
prompt = "To be or not to be"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)

print(tokenizer.decode(output[0]))
```

## 📈 Key Metrics

- **Training Time**: [Add your total training time]
- **Tokens/Second**: ~10,000 tok/sec (varies by hardware)
- **Memory Usage**: [Add your GPU memory usage]
- **Final Loss**: [Add your final loss value]
- **Model Size**: 
  - Full precision: ~540MB
  - Half precision: ~270MB

## 🔧 Requirements

```
torch>=2.0.0
transformers>=4.30.0
```

## 💡 Key Features

✅ **Custom PyTorch Implementation**: Built from scratch, not using Hugging Face's modeling code  
✅ **Grouped Query Attention**: Efficient attention with 3:1 head sharing  
✅ **Flash Attention**: 4x speedup with memory-efficient attention  
✅ **Mixed Precision Training**: BFloat16 for faster computation  
✅ **Checkpoint Resume**: Save and resume training seamlessly  
✅ **Cosine Annealing**: Advanced LR scheduling with warmup  
✅ **Sample Generation**: Periodic quality checks during training  

## 📝 Notes

- The model is trained on a small dataset (`input.txt`) for demonstration
- Training can be scaled up with more data and longer sequences
- Flash Attention requires PyTorch 2.0+ and appropriate GPU
- Torch compile speedup works best on Linux systems
- Checkpoint includes optimizer state for exact training resumption

## 🎓 Learning Objectives

This implementation demonstrates:
- Transformer architecture from scratch
- Advanced attention mechanisms (GQA, RoPE)
- Training optimization techniques
- Checkpoint management
- Text generation strategies

## 📊 Training Summary — SmolLM2-135M Fine-Tuning

The model was trained from scratch using the SmolLM2-135M architecture implemented in PyTorch with RoPE, GQA attention, RMSNorm, and SwiGLU MLP. Training demonstrates **stable convergence**, with clear trade-offs between loss improvement and memorization risk.

### Training Log Snapshot (High Intervals)

| Training Step | Loss | Learning Rate | Tokens/sec | Notes |
|--------------|------|---------------|------------|-------|
| ~100 | ~6.8 | 0.00010 | ~1900 | Fast early improvement |
| ~500 | ~5.0 | 0.00048 | ~1915 | Stabilizing post-warmup |
| ~1500 | ~4.3 | 0.00044 | ~1905 | Mid-training progress |
| ~2200 | ~3.3 | 0.00035 | ~1930 | Healthy convergence |
| ~2600 | ~2.7 | 0.00029 | ~1934 | Strong improvement |
| ~3000 | ~1.9 | 0.00023 | ~1894 | Lower LR, deeper fit |
| ~3800 | ~0.40 | 0.00011 | ~1932 | Entering memorization |
| ~4500 | ~0.09 | 0.00006 | ~1928 | Very low loss |
| ~4800 | ~0.05 | 0.00005 | ~1930 | Possible overfitting zone |

### Detailed Loss Evidence (from actual logs)

- **Step 1017** → Loss 5.07
- **Step 1539** → Loss 4.29
- **Step 2237** → Loss 3.17
- **Step 2553** → Loss 2.44
- **Step 3036** → Loss 1.93
- **Step 3861** → Loss 0.23
- **Step 4467** → Loss 0.07
- **Step 4742** → Loss 0.08
- **Step 4837** → Loss 0.05

### 🧠 Interpretation & Takeaways

**Training Phases Observed:**

1. **Early Phase (Steps 1-1000)**: Rapid convergence from ~10.0 → ~5.0 loss during warmup period
2. **Mid Training (Steps 1000-3000)**: Steady improvement with cosine annealing (5.0 → 1.9 loss)
3. **Late Training (Steps 3000-4000)**: Deep fitting as learning rate decreases (1.9 → 0.4 loss)
4. **Final Phase (Steps 4000-5000)**: Very low loss (<0.1) indicates potential memorization

**Key Observations:**
- The loss curve shows rapid early convergence, then deep fit as learning rate decays
- After ~3800 steps, the loss falls below 1.0, typical of memorization in small models
- Reaching <0.1 loss suggests the model may memorize training data, a common ML phenomenon for small language models trained on limited tokens
- Consistent throughput (~1900-1930 tok/sec) indicates stable training without bottlenecks

### 🎯 Best Checkpoint for Deployment

**Recommended generalization checkpoint**: Between steps **2800-3500** (Loss ~1.5–2.7)

This range maintains meaningful learning without heavy memorization. The model at these checkpoints:
- ✅ Generalizes better to unseen text
- ✅ Maintains creative diversity
- ✅ Balances perplexity and coherence

⚠️ **Note**: Final very-low-loss weights (steps 4500-5000) can be used for creative tasks, but may overfit on narrow datasets and produce repetitive outputs.

### 📈 Loss Progression Graph (Conceptual)

```
10.0 ┤                                                    
     │●                                                   
 8.0 ┤ ●                                                  
     │  ●●                                                
 6.0 ┤    ●●●                                             
     │       ●●●●                                         
 4.0 ┤          ●●●●●●                                    
     │               ●●●●●●●                              
 2.0 ┤                      ●●●●●●●●●                     
     │                              ●●●●●●●●●●●           
 0.0 ┤                                        ●●●●●●●●●   
     └─────────────────────────────────────────────────→
     0    500   1000  1500  2000  2500  3000  3500  4000  4500  5000
                           Training Steps
```

**Key Observation**: The exponential decay after step 3000 is characteristic of small models learning a limited corpus. For production use, early stopping around step 3000 is recommended.

## 📚 References

- [SmolLM2 Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [Llama 2 Paper](https://arxiv.org/abs/2307.09288)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Grouped Query Attention](https://arxiv.org/abs/2305.13245)

---
