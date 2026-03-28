"""Test loading the smoothed model from HuggingFace Hub and run a simple generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "jsyeom/llama-2-13b-hf-smooth"

print(f"Loading model from: {REPO_ID}")
model = AutoModelForCausalLM.from_pretrained(
    REPO_ID, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

print(f"Model type: {type(model).__name__}")
print(f"Model dtype: {model.dtype}")
print(f"Num parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# Generation test
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"\nPrompt: {prompt}")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated}")

print("\nTest passed.")
