import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "models/gemma-7b-it-2bit-custom"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prompts to run
    prompts = [
        "Solve this problem: (15 × 8) + (72 ÷ 6) - 13 =",
        "Name the author who wrote 'Game of Thrones' and their height.",
        "Arrange these steps in order for making coffee: Grind beans, Boil water, Pour water, Add beans to filter.",
        "What is love?",
        "Write a Java function that reverses a string without using built-in reverse methods.",
        "If on the day my sister is born im double her age, how old am i when she is 50?",
        "Complete the analogy: Thermometer is to temperature as barometer is to ______.",
        "Does this statement contain a contradiction: 'The silent orchestra played loudly all night'?",
        "If all humans are mortal, and Socrates is human, what can we conclude about Socrates?",
        "Write a short paragraph describing a sunset over the ocean, using vivid sensory details."
    ]

    # Generate and print Q/A
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device if device != "cpu" else "cpu")
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # greedy
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            # Decode only the new tokens
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"Q: {prompt}")
            print(f"A: {answer}\n")

if __name__ == "__main__":
    main()
