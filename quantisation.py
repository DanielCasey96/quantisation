import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

def check_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def load_model_quantized(model_path, quantization_level="none"):
    print(f"Loading model with {quantization_level} quantization...")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if quantization_level == "2bit":
        # Try to load pre-quantized 2-bit model, fallback to custom
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "models/mistral-7b-2bit-custom",
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✅ Loaded pre-quantized 2-bit model")
            return model
        except:
            print("Pre-quantized 2-bit not found, using custom 2-bit")
            return load_custom_2bit(model_path)

    elif quantization_level == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization_level == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    else:  # "none" - 16bit
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model

def load_custom_2bit(model_path):
    """Apply custom 2-bit quantization on the fly"""
    print("Applying custom 2-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Apply simple 2-bit quantization to linear layers
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

    model.eval()
    return model

def test_memory_usage(model, model_name):
    memory_allocated = 0
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{model_name} - Memory allocated: {memory_allocated:.2f} GB")
        print(f"{model_name} - Memory reserved: {memory_reserved:.2f} GB")
    else:
        print("CUDA not available - running on CPU")
    return memory_allocated

def test_simple_inference(model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    prompts = [
        "Solve step by step: (15 × 8) + (72 ÷ 6) - 13 =",
        "Name the author who wrote 'One Hundred Years of Solitude' and their country of origin.",
        "Arrange these steps in order for making coffee: Grind beans, Boil water, Pour water, Add beans to filter.",
        "Explain the subtle difference between 'happy,' 'joyful,' and 'ecstatic' with examples.",
        "Write a Python function that reverses a string without using built-in reverse methods.",
        "If today is March 15, 2024, what day of the week will April 10, 2024 be?",
        "Complete the analogy: Thermometer is to temperature as barometer is to ______.",
        "Does this statement contain a contradiction: 'The silent orchestra played loudly all night'?",
        "If all humans are mortal, and Socrates is human, what can we conclude about Socrates?",
        "Write a short paragraph describing a sunset over the ocean, using vivid sensory details."
    ]
    responses = []
    with torch.inference_mode():
        for input_text in prompts:
            inputs = tokenizer(input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            responses.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return responses

if __name__ == "__main__":
    # Test with Mistral since we have 2-bit for it
    model_path = "models/mistral-7b"

    print("System Check")
    check_cuda()

    print("\nTesting Quantization Levels")

    # Test all quantization levels including 2-bit
    for quant_level in ["2bit", "4bit", "8bit", "none"]:
        print(f"\n=== Testing {quant_level} quantization ===")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = load_model_quantized(model_path, quant_level)
            memory_used = test_memory_usage(model, f"Mistral-7B ({quant_level})")

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

            responses = test_simple_inference(model, tokenizer)
            print(f"Generated {len(responses)} responses")

            # Print first response as sample
            if responses:
                print(f"Sample response: {responses[0][:200]}...")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with {quant_level}: {e}")
            import traceback
            traceback.print_exc()