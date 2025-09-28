import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IN = "models/mistral-7b"
MODEL_OUT = "models/mistral-7b-3bit-custom"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVELS = 8          # 3-bit
EPS = 1e-8
CLIP_PCT = 0.999    # percentile clipping (keep central 99.9%)
SKIP_MODULE_SUBSTR = [
    "embed_tokens",
    "lm_head",
    "norm",
    "layernorm",
    "LayerNorm"
]

def should_skip(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in SKIP_MODULE_SUBSTR)

def per_channel_stats(w: torch.Tensor):
    # w: [out, in]
    # Percentile clipping to reduce outlier impact
    sorted_w, _ = torch.sort(w, dim=1)
    n = w.size(1)
    hi_idx = int((CLIP_PCT) * (n - 1))
    lo_idx = int((1 - CLIP_PCT) * 0)  # effectively 0
    hi = sorted_w[:, hi_idx:hi_idx+1]
    lo = sorted_w[:, lo_idx:lo_idx+1]
    return lo, hi

def quantize_per_channel_symmetric(w: torch.Tensor):
    wf = w.float()
    lo, hi = per_channel_stats(wf)
    max_abs = torch.max(hi.abs(), lo.abs()).clamp_min(EPS)
    scale = max_abs / ((LEVELS - 1) / 2)  # symmetric signed range
    q = torch.round(wf / scale).clamp(-(LEVELS//2), (LEVELS//2))
    deq = q * scale
    return deq.to(w.dtype)

def quantize_linear(name: str, linear: nn.Linear):
    if getattr(linear.weight, "is_meta", False):
        print(f"[skip meta] {name}")
        return
    w = linear.weight.data
    w_q = quantize_per_channel_symmetric(w)
    if torch.isnan(w_q).any() or torch.isinf(w_q).any():
        print(f"[revert NaN] {name}")
        return
    linear.weight.data = w_q

def main():
    print("Loading model (full, no meta)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IN,
        dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=False
    )
    model.to(DEVICE)
    model.eval()
    model.config.use_cache = False  # just to reassure no reuse

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Quantizing (skipping embeddings/lm_head/norms)...")
    total = quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total += 1
            if should_skip(name):
                continue
            quantize_linear(name, module)
            quantized += 1
    print(f"Linear layers seen: {total}, quantized: {quantized}")

    # Sanity check
    with torch.no_grad():
        inputs = tokenizer("Quality check.", return_tensors="pt").to(DEVICE)
        logits = model(**inputs).logits[:, -1, :]
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("NaN/Inf logits post-quantization.")
        probs = torch.softmax(logits.float(), dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        print(f"Sanity top token prob: {top_prob.item():.4f}")

    print(f"Saving -> {MODEL_OUT}")
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print("Done.")

if __name__ == "__main__":
    main()
