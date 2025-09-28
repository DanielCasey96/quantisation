import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IN = "models/gemma-7b-it"
MODEL_OUT = "models/gemma-7b-it-2bit-custom"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = 4  # 2-bit
EPS = 1e-8
SKIP_LM_HEAD_FIRST_PASS = True  # set False after validating core layers

def quantize_per_channel(weight: torch.Tensor, levels: int = LEVELS):
    # Shape: [out_features, in_features]
    w = weight.float()
    min_vals = w.min(dim=1, keepdim=True).values
    max_vals = w.max(dim=1, keepdim=True).values
    spans = (max_vals - min_vals).clamp_min(EPS)
    scale = spans / (levels - 1)
    zero_point = torch.round(-min_vals / scale)
    q = torch.round(w / scale + zero_point)
    q = q.clamp(0, levels - 1)
    deq = (q - zero_point) * scale
    return deq.to(weight.dtype)

def safe_quantize_linear(module_name: str, linear: nn.Linear):
    w = linear.weight.data
    if getattr(w, "is_meta", False):
        print(f"[skip meta] {module_name}")
        return
    w_q = quantize_per_channel(w)
    if torch.isnan(w_q).any() or torch.isinf(w_q).any():
        print(f"[revert NaN] {module_name}")
        return
    linear.weight.data = w_q

def main():
    print("Loading full model (no meta tensors)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IN,
        dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=False
    )
    model.eval()
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Quantizing Linear layers (per-channel 2-bit)...")
    total = quantized = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if SKIP_LM_HEAD_FIRST_PASS and name.endswith("lm_head"):
                continue
            total += 1
            safe_quantize_linear(name, mod)
            quantized += 1
    print(f"Linear layers visited: {total}, quantized: {quantized}")

    # Sanity forward pass
    test_text = "Test."
    inputs = tokenizer(test_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
        last = out.logits[:, -1, :]
        if torch.isnan(last).any() or torch.isinf(last).any():
            raise RuntimeError("Found NaN/Inf in logits after quantization.")
    print("Sanity forward pass OK.")

    print(f"Saving to {MODEL_OUT} ...")
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print("Done.")

if __name__ == "__main__":
    main()
