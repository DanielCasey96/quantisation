from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch

model_path = "models/llama2-7b-chat"
save_path = "models/llama2-7b-chat-2bit"

# First, let's load the model with transformers to check the config
print("Loading model to check configuration...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print(f"Model config: {model.config}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Num attention heads: {model.config.num_attention_heads}")
print(f"Head dimension: {model.config.hidden_size // model.config.num_attention_heads}")

del model  # Free memory

# Now set up quantization with proper configuration
quantize_config = BaseQuantizeConfig(
    bits=2,
    group_size=128,
    desc_act=False,
)

print("Loading model for quantization...")
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create better calibration data - use shorter sequences to avoid RoPE issues
calib_texts = []
for i in range(32):  # More samples but shorter
    calib_texts.extend([
        f"Sample text {i} for calibration purposes.",
        f"This is calibration example number {i}.",
        f"Machine learning model quantization example {i}.",
        f"Text sequence for model calibration {i}.",
    ])

print("Tokenizing calibration data...")
enc = tokenizer(
    calib_texts,
    padding=True,
    truncation=True,
    max_length=512,  # Shorter sequences to avoid dimension issues
    return_tensors="pt",
)

examples = [
    {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
    for i in range(len(calib_texts))
]

print("Starting quantization...")
try:
    model.quantize(
        examples=examples,
        batch_size=1,
        use_triton=False,
    )

    print("Saving quantized model...")
    model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"2-bit quantization complete! Model saved to: {save_path}")

except Exception as e:
    print(f"Quantization failed: {e}")
    print("Trying alternative approach...")

    # Alternative: Try with even simpler calibration
    print("Attempting with minimal calibration...")
    dummy_data = ["The quick brown fox jumps over the lazy dog. " * 10]
    enc = tokenizer(dummy_data, return_tensors="pt", max_length=256)
    examples = [{"input_ids": enc["input_ids"][0]}]

    try:
        model.quantize(examples=examples, batch_size=1, use_triton=False)
        model.save_quantized(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Alternative quantization successful! Model saved to: {save_path}")
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")