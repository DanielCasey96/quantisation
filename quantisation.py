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

    if quantization_level == "8bit":
        if not torch.cuda.is_available():
            raise Exception("8-bit quantization requires CUDA")

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offloading
        )
    elif quantization_level == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    return model

def test_memory_usage(model, model_name):
    memory_allocated = 0
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        print(f"{model_name} - Memory allocated: {memory_allocated:.2f} GB")
        print(f"{model_name} - Memory reserved: {memory_reserved:.2f} GB")
    else:
        print("CUDA not available - running on CPU")

    return memory_allocated

def test_simple_inference(model, tokenizer):
    input_text = "The capital of France is"
    inputs = tokenizer(input_text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=20, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    model_path = "models/mistral-7b"
    # Gemma downloaded, waiting on Meta approval

    print("System Check")
    check_cuda()

    print("\nTesting Quantization Levels")

    for quant_level in ["none", "8bit", "4bit"]:
        print(f"\nTesting {quant_level} quantization ")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = load_model_quantized(model_path, quant_level)
            memory_used = test_memory_usage(model, f"Mistral-7B ({quant_level})")

            tokenizer = AutoTokenizer.from_pretrained(model_path)

            response = test_simple_inference(model, tokenizer)
            print(f"Test response: {response}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with {quant_level}: {e}")
            import traceback
            traceback.print_exc()