import gradio as gr
import torch
import torch.nn.functional as F
import os
from transformers import AutoTokenizer
from s13_bikash_smollm2_135m import SmolLM2Config, SmolLM2ForCausalLM

# ---------------- CONFIG + CHECKPOINT -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ckpt_path = "model_final.pth"   # your smolLM checkpoint
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f"Checkpoint '{ckpt_path}' not found. Upload it next to this file."
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Init model architecture
config = SmolLM2Config()
model = SmolLM2ForCausalLM(config)

# ---------------- LOAD CHECKPOINT WITH KEY FIX -----------------
state_dict = torch.load(ckpt_path, map_location=device)

# Rename mismatched weight keys (Colab → Inference)
new_state_dict = {}
for k, v in state_dict.items():
    k2 = k
    k2 = k2.replace("embed_tokens", "embed")
    k2 = k2.replace("input_layernorm", "input_norm")
    k2 = k2.replace("post_attention_layernorm", "post_norm")
    new_state_dict[k2] = v

state_dict = new_state_dict

# Load weights (ignore missing LM head bias safely)
model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

# ---------------- TEXT GENERATION FUNCTION -----------------
def generate_text(prompt, max_tokens=50, temperature=0.8, top_k=50):
    if not prompt.strip():
        return "Please enter a prompt."

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=None,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ---------------- GRADIO UI (UNCHANGED) -----------------
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", value="Once upon a time", lines=3),
        gr.Slider(10, 200, value=50, step=10, label="Max new tokens"),
        gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(0, 100, value=50, step=10, label="Top-K sampling"),
    ],
    outputs=gr.Textbox(label="Generated text", lines=10),
    title="SmolLM2-135M Model",
    description="Inference demo using trained SmolLM2-135M checkpoint.",
    examples=[
        ["CORIOLANUS: What is the city but the people?"],
        ["MENENIUS: There was a time when all the body's members rebelled against the belly—"],
        ["CITIZEN: We have power in ourselves to do it, but it is a power that we have no power to do."],
        ["O pride, thou noble fault that ruins noble men!"],
        ["War’s trumpet sounds, and peace withdraws her hand."]
    ],
    examples_per_page=5,
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
