import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/llama2-7b-chat"
save_path = "models/llama2-7b-chat-2bit-custom"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Applying custom 2-bit quantization...")
def apply_2bit_quantization(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                weights = module.weight.data
                # 2-bit quantization: 4 levels (0, 1, 2, 3)
                min_val = weights.min()
                max_val = weights.max()
                scale = (max_val - min_val) / 3  # 2-bit has 4 values: 0,1,2,3

                # Quantize to nearest integer
                quantized = torch.round((weights - min_val) / scale)
                quantized = torch.clamp(quantized, 0, 3)

                # Dequantize back for inference
                module.weight.data = quantized * scale + min_val
    return model

model = apply_2bit_quantization(model)

print("Saving model...")
model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(save_path)

print("✅ Custom 2-bit quantization complete!")
print(f"Model saved to: {save_path}")