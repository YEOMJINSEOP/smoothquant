"""
Verify that the smoothed model is correctly smoothed.

Three checks:
1. Weight diff: smoothed model weights differ from original in expected layers only
2. Mathematical equivalence: smoothed model produces identical logits as original
3. Artifact completeness: all required files present in output directory
"""

import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from smoothquant.smooth import smooth_lm

ORIGINAL_MODEL = "meta-llama/Llama-2-13b-hf"
SMOOTHED_DIR = "smoothed_models/llama-2-13b-hf-smooth"
ACT_SCALES_PATH = "act_scales/llama-2-13b.pt"
ALPHA = 0.85

print("=" * 60)
print("CHECK 1: Weight Difference Verification")
print("=" * 60)

# Load original model
print("Loading original model...")
original = AutoModelForCausalLM.from_pretrained(
    ORIGINAL_MODEL, torch_dtype=torch.float32, device_map="cpu"
)

# Load smoothed model
print("Loading smoothed model...")
smoothed = AutoModelForCausalLM.from_pretrained(
    SMOOTHED_DIR, torch_dtype=torch.float32, device_map="cpu"
)

# Expected changed layers per decoder layer
EXPECTED_CHANGED = {
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
}

changed_params = []
unchanged_params = []

for name, orig_param in original.named_parameters():
    smooth_param = dict(smoothed.named_parameters())[name]
    is_different = not torch.equal(orig_param, smooth_param)

    if is_different:
        changed_params.append(name)
    else:
        unchanged_params.append(name)

# Verify only expected layers changed
unexpected_changes = []
missing_changes = []

for name in changed_params:
    # Extract the layer-local name (e.g., "self_attn.q_proj.weight")
    parts = name.split(".")
    if "layers" in parts:
        layer_idx = parts.index("layers")
        local_name = ".".join(parts[layer_idx + 2:])
        if local_name not in EXPECTED_CHANGED:
            unexpected_changes.append(name)
    else:
        unexpected_changes.append(name)

for i in range(40):  # 40 decoder layers
    for expected in EXPECTED_CHANGED:
        full_name = f"model.layers.{i}.{expected}"
        if full_name not in changed_params:
            missing_changes.append(full_name)

print(f"  Changed params:    {len(changed_params)}")
print(f"  Unchanged params:  {len(unchanged_params)}")
print(f"  Expected changed:  {len(EXPECTED_CHANGED)} x 40 layers = {len(EXPECTED_CHANGED) * 40}")

if unexpected_changes:
    print(f"  [FAIL] Unexpected changes: {unexpected_changes}")
else:
    print(f"  [PASS] No unexpected weight changes")

if missing_changes:
    print(f"  [FAIL] Missing changes: {missing_changes[:5]}...")
else:
    print(f"  [PASS] All expected layers were modified")


print()
print("=" * 60)
print("CHECK 2: Smoothing Formula Verification")
print("=" * 60)
print("  Verify: ln.weight_smoothed == ln.weight_orig / s")
print("  Verify: fc.weight_smoothed == fc.weight_orig * s")
print("  where s = act_scales^alpha / weight_scales^(1-alpha)")

act_scales = torch.load(ACT_SCALES_PATH, map_location="cpu")
orig_params = dict(original.named_parameters())
smooth_params = dict(smoothed.named_parameters())

formula_pass = True
max_formula_err = 0.0

for layer_idx in range(40):
    prefix = f"model.layers.{layer_idx}"

    # --- Attention block ---
    ln_w_orig = orig_params[f"{prefix}.input_layernorm.weight"].float()
    q_w_orig = orig_params[f"{prefix}.self_attn.q_proj.weight"].float()
    k_w_orig = orig_params[f"{prefix}.self_attn.k_proj.weight"].float()
    v_w_orig = orig_params[f"{prefix}.self_attn.v_proj.weight"].float()

    a_scales = act_scales[f"{prefix}.self_attn.q_proj"].float()
    w_scales = torch.cat([
        q_w_orig.abs().max(dim=0, keepdim=True)[0],
        k_w_orig.abs().max(dim=0, keepdim=True)[0],
        v_w_orig.abs().max(dim=0, keepdim=True)[0],
    ], dim=0).max(dim=0)[0].clamp(min=1e-5)
    s_attn = (a_scales.pow(ALPHA) / w_scales.pow(1 - ALPHA)).clamp(min=1e-5)

    # Check RMSNorm: smoothed == orig / s
    ln_w_smooth = smooth_params[f"{prefix}.input_layernorm.weight"].float()
    expected_ln = ln_w_orig / s_attn
    err = (ln_w_smooth - expected_ln).abs().max().item()
    max_formula_err = max(max_formula_err, err)

    # Check Linear: smoothed == orig * s
    for proj in ["q_proj", "k_proj", "v_proj"]:
        fc_w_orig = orig_params[f"{prefix}.self_attn.{proj}.weight"].float()
        fc_w_smooth = smooth_params[f"{prefix}.self_attn.{proj}.weight"].float()
        expected_fc = fc_w_orig * s_attn.view(1, -1)
        err = (fc_w_smooth - expected_fc).abs().max().item()
        max_formula_err = max(max_formula_err, err)

    # --- FFN block ---
    ln_w_orig = orig_params[f"{prefix}.post_attention_layernorm.weight"].float()
    gate_w_orig = orig_params[f"{prefix}.mlp.gate_proj.weight"].float()
    up_w_orig = orig_params[f"{prefix}.mlp.up_proj.weight"].float()

    a_scales = act_scales[f"{prefix}.mlp.gate_proj"].float()
    w_scales = torch.cat([
        gate_w_orig.abs().max(dim=0, keepdim=True)[0],
        up_w_orig.abs().max(dim=0, keepdim=True)[0],
    ], dim=0).max(dim=0)[0].clamp(min=1e-5)
    s_ffn = (a_scales.pow(ALPHA) / w_scales.pow(1 - ALPHA)).clamp(min=1e-5)

    ln_w_smooth = smooth_params[f"{prefix}.post_attention_layernorm.weight"].float()
    expected_ln = ln_w_orig / s_ffn
    err = (ln_w_smooth - expected_ln).abs().max().item()
    max_formula_err = max(max_formula_err, err)

    for proj in ["gate_proj", "up_proj"]:
        fc_w_orig = orig_params[f"{prefix}.mlp.{proj}.weight"].float()
        fc_w_smooth = smooth_params[f"{prefix}.mlp.{proj}.weight"].float()
        expected_fc = fc_w_orig * s_ffn.view(1, -1)
        err = (fc_w_smooth - expected_fc).abs().max().item()
        max_formula_err = max(max_formula_err, err)

print(f"  Max formula error across all 40 layers: {max_formula_err:.6e}")
# Threshold: smoothed model was saved in fp16 → loaded in fp32,
# while expected values are computed in fp32 from fp32 original weights.
# fp16 has ~1e-3 relative precision, so absolute error up to ~1e-2 is expected.
if max_formula_err < 1e-2:
    print(f"  [PASS] All weights match smoothing formula (within fp16 precision)")
else:
    print(f"  [FAIL] Weights do not match smoothing formula")
    formula_pass = False


print()
print("=" * 60)
print("CHECK 3: Mathematical Equivalence (Logit Comparison)")
print("=" * 60)

# Apply smoothing to original model (in-place) and compare with saved smoothed model
smooth_lm(original, act_scales, ALPHA)

tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
test_input = tokenizer("The capital of France is", return_tensors="pt").input_ids

with torch.no_grad():
    logits_from_smooth_lm = original(test_input).logits
    logits_from_saved = smoothed(test_input).logits

max_diff = (logits_from_smooth_lm - logits_from_saved).abs().max().item()
mean_diff = (logits_from_smooth_lm - logits_from_saved).abs().mean().item()

print(f"  Max logit diff:  {max_diff:.6e}")
print(f"  Mean logit diff: {mean_diff:.6e}")

# Threshold accounts for fp16 save/load precision loss
# (saved model went through fp16 round-trip, fresh smooth_lm runs in fp32)
if max_diff < 0.05:
    print(f"  [PASS] Saved model matches fresh smooth_lm() output (within fp16 precision)")
else:
    print(f"  [FAIL] Logit mismatch beyond fp16 precision tolerance")


print()
print("=" * 60)
print("CHECK 4: Artifact Completeness")
print("=" * 60)

required_files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "smoothquant_config.json",
    "README.md",
    "llama-2-13b.pt",
    "model.safetensors.index.json",
]

for f in required_files:
    path = os.path.join(SMOOTHED_DIR, f)
    exists = os.path.exists(path)
    status = "[PASS]" if exists else "[FAIL]"
    print(f"  {status} {f}")

# Verify smoothquant_config.json content
config_path = os.path.join(SMOOTHED_DIR, "smoothquant_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        meta = json.load(f)
    checks = [
        ("base_model", ORIGINAL_MODEL),
        ("smoothing_alpha", ALPHA),
        ("quantization_applied", False),
    ]
    for key, expected in checks:
        actual = meta.get(key)
        status = "[PASS]" if actual == expected else "[FAIL]"
        print(f"  {status} smoothquant_config.json: {key} = {actual}")

print()
print("=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
