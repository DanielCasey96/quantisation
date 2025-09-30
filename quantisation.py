import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import time

def check_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def load_model_quantized(model_path, quantization_level="none"):
    print(f"Loading model with {quantization_level} quantization")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if quantization_level == "8bit":
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
        device_map="cuda",
        torch_dtype=dtype,
        trust_remote_code=True,
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

def get_comprehensive_prompts():
    """Returns 60 prompts across 6 categories with 10 prompts each"""

    categories = {
        "mathematical_reasoning": [
            "Solve this problem with a single number output: (15 × 8) + (72 ÷ 6) - 13 =",
            "Calculate: 2^8 + 3^4 - 100 =",
            "What is 25% of 840?",
            "If x + 2x = 45, what is the value of x?",
            "Solve: (18 × 3) ÷ (12 - 6) =",
            "What is the square root of 256?",
            "Calculate the area of a circle with radius 7 (use π ≈ 3.14):",
            "If a car travels 60 miles in 1.5 hours, what is its speed in mph?",
            "What is 3/4 of 200?",
            "Solve: 5² + 12² - 8² ="
        ],

        "factual_knowledge": [
            "Name the author who wrote 'Game of Thrones' and their height.",
            "Who invented the telephone and in what year?",
            "What is the capital of Brazil and its population (approximately)?",
            "Name the director of the movie 'Inception' and their nationality.",
            "Who discovered penicillin and what year did they win the Nobel Prize?",
            "What is the chemical symbol for gold and its atomic number?",
            "Name the CEO of Tesla and their year of birth.",
            "What is the largest planet in our solar system and its diameter?",
            "Who wrote '1984' and what year was it published?",
            "What is the height of Mount Everest in meters and feet?"
        ],

        "logical_sequencing": [
            "Arrange these steps in order for making coffee: Grind beans, Boil water, Pour water, Add beans to filter.",
            "Arrange these steps for baking bread: Preheat oven, Mix ingredients, Knead dough, Let dough rise.",
            "Order these steps for sending an email: Write content, Add recipient, Click send, Add subject.",
            "Sequence these programming steps: Write code, Test functionality, Debug errors, Plan algorithm.",
            "Arrange these morning routine steps: Brush teeth, Eat breakfast, Shower, Get dressed.",
            "Order these book writing steps: Write draft, Edit manuscript, Publish book, Outline chapters.",
            "Sequence these car maintenance steps: Change oil, Check tires, Wash car, Inspect brakes.",
            "Arrange these gardening steps: Plant seeds, Water plants, Prepare soil, Harvest crops.",
            "Order these travel planning steps: Book flights, Pack bags, Research destination, Get passport.",
            "Sequence these problem-solving steps: Identify problem, Implement solution, Analyze causes, Evaluate results."
        ],

        "code_generation": [
            "Write a Java function that reverses a string without using built-in reverse methods.",
            "Write a Python function to check if a number is prime.",
            "Create a JavaScript function that finds the maximum number in an array.",
            "Write a C++ function to calculate the factorial of a number recursively.",
            "Create a Python class for a BankAccount with deposit and withdraw methods.",
            "Write a Java method to sort an integer array using bubble sort.",
            "Create a JavaScript function that removes duplicates from an array.",
            "Write a Python function to count the frequency of words in a string.",
            "Create a C++ program that reads a file and counts the lines.",
            "Write a Java function to check if two strings are anagrams."
        ],

        "conceptual_understanding": [
            "What is love?",
            "Explain the concept of artificial intelligence in simple terms.",
            "What is climate change and its main causes?",
            "Explain how photosynthesis works.",
            "What is the difference between weather and climate?",
            "Explain the concept of supply and demand in economics.",
            "What is blockchain technology and how does it work?",
            "Explain the theory of relativity in simple terms.",
            "What is machine learning and how is it different from traditional programming?",
            "Explain what causes the seasons on Earth."
        ],

        "analogical_reasoning": [
            "Complete the analogy: Thermometer is to temperature as barometer is to ______.",
            "Complete: Keyboard is to type as mouse is to ______.",
            "Complete: Doctor is to hospital as teacher is to ______.",
            "Complete: Author is to book as composer is to ______.",
            "Complete: Fish is to water as bird is to ______.",
            "Complete: Wheel is to car as wing is to ______.",
            "Complete: Pen is to write as knife is to ______.",
            "Complete: Solar is to sun as lunar is to ______.",
            "Complete: Microscope is to small as telescope is to ______.",
            "Complete: Chef is to kitchen as architect is to ______."
        ]
    }

    # Flatten into single list while keeping track of categories
    all_prompts = []
    for category, prompts in categories.items():
        for prompt in prompts:
            all_prompts.append((category, prompt))

    return all_prompts

def model_setup(model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    prompts_with_categories = get_comprehensive_prompts()

    responses_by_category = {category: [] for category in set(cat for cat, _ in prompts_with_categories)}

    print(f"Running {len(prompts_with_categories)} prompts across {len(responses_by_category)} categories...")

    with torch.inference_mode():
        for i, (category, input_text) in enumerate(prompts_with_categories):
            start_time = time.time()

            inputs = tokenizer(input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=300,  # Reduced for efficiency with more prompts
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                temperature=0.1,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses_by_category[category].append((input_text, response))

            elapsed = time.time() - start_time
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{len(prompts_with_categories)} prompts (Category: {category}, Time: {elapsed:.2f}s)")

    return responses_by_category

def save_responses(responses_by_category, model_name, quantization_level):
    """Save responses to files for analysis"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"results/{model_name}_{quantization_level}_{timestamp}.txt"

    os.makedirs("results", exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Quantization: {quantization_level}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        for category, responses in responses_by_category.items():
            f.write(f"CATEGORY: {category.upper()}\n")
            f.write("-" * 40 + "\n")
            for i, (prompt, response) in enumerate(responses, 1):
                f.write(f"Prompt {i}: {prompt}\n")
                f.write(f"Response: {response}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")

    print(f"Results saved to: {filename}")
    return filename

if __name__ == "__main__":
    model_path = "models/mistral-7b"

    print("System Check")
    check_cuda()

    print(f"\nTesting with comprehensive prompt set (60 prompts across 6 categories)")

    for quant_level in ["none", "8bit", "4bit"]:
        print(f"\n{'='*60}")
        print(f"Testing {quant_level} quantization")
        print(f"{'='*60}")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start_time = time.time()
            model = load_model_quantized(model_path, quant_level)
            memory_used = test_memory_usage(model, f"mistral-7b ({quant_level})")

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

            responses_by_category = model_setup(model, tokenizer)

            # Print summary
            total_responses = sum(len(responses) for responses in responses_by_category.values())
            print(f"\nGenerated {total_responses} total responses across categories:")
            for category, responses in responses_by_category.items():
                print(f"  {category}: {len(responses)} responses")

            # Save to file
            save_responses(responses_by_category, "mistral-7b", quant_level)

            total_time = time.time() - start_time
            print(f"Total time for {quant_level}: {total_time:.2f} seconds")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with {quant_level}: {e}")
            import traceback
            traceback.print_exc()