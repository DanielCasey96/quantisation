import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_2bit_model():
    print("=== Testing Pre-Quantized 2-bit Model ===")

    # Check GPU status
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load the pre-quantized 2-bit model - FORCE GPU
    model_path = "models/mistral-7b-2bit-custom"

    print("Loading 2-bit model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "auto"  # Force specific GPU
    )

    # Move model to GPU explicitly
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"Model device: {next(model.parameters()).device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Memory usage before
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nMemory before: {memory_before:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

    # Test with a simple prompt
    print("\n--- Testing model capability ---")
    test_prompt = "The capital of France is"

    inputs = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}  # Force inputs to GPU

    print(f"Input device: {inputs['input_ids'].device}")
    print(f"Model device: {next(model.parameters()).device}")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"Prompt: '{test_prompt}'")
    print(f"Full response: '{full_response}'")
    print(f"New text only: '{new_text}'")

    # Memory usage after
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated() / 1024**3
        print(f"\nMemory after: {memory_after:.2f} GB allocated")
        print(f"Memory delta: {memory_after - memory_before:.2f} GB")

    # Check if model works
    # if new_text.strip() == "" or test_prompt in new_text:
    #     print("\n❌ MODEL IS BROKEN - 2-bit quantization destroyed the model")
    #     return

    # Run full test if model works
    print("\n--- Running full test suite ---")
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
        for i, input_text in enumerate(prompts, 1):
            print(f"Prompt {i}/10...")

            inputs = tokenizer(input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)

    print("\n" + "="*50)
    print("2-BIT MODEL RESULTS:")
    print("="*50)

    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"\n--- Prompt {i} ---")
        print(f"Q: {prompt}")
        print(f"A: {response}")
        print("-" * 80)

if __name__ == "__main__":
    test_2bit_model()