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

    quantization_config = None
    if quantization_level == "8bit":
        # Keep the whole model on GPU; no CPU offload
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization_level == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="cuda" if torch.cuda.is_available() else "cpu",  # avoid CPU offload
        dtype=dtype,  # replaces deprecated torch_dtype
        trust_remote_code=True,
        # Uncomment if flash-attn 2 is installed:
        # attn_implementation="flash_attention_2",
    )
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
    model_path = "models/llama2-7b-chat"

    print("System Check")
    check_cuda()

    print("\nTesting Quantization Levels")

    # On a 16 GB GPU, prefer 4bit or 8bit to keep everything on-GPU.
    for quant_level in ["4bit", "8bit", "none"]:
        print(f"\nTesting {quant_level} quantization ")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = load_model_quantized(model_path, quant_level)
            memory_used = test_memory_usage(model, f"Gemma-7B ({quant_level})")

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

            response = test_simple_inference(model, tokenizer)
            print(f"Test response: {response}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with {quant_level}: {e}")
            import traceback
            traceback.print_exc()
