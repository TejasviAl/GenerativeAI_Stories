# 5_unique_stories_generator_same_prompt.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the model and tokenizer
model_path = "./story_model"  # Replace with your trained model path
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Make sure padding token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_stories(prompt, max_length=200, num_return_sequences=5, temperature=0.8, top_p=0.9, no_repeat_ngram_size=3):
    """
    Generates multiple story variations for the same prompt.
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate multiple outputs with sampling
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,              # enable sampling
        temperature=temperature,     # creativity level
        top_p=top_p,                 # nucleus sampling
        no_repeat_ngram_size=no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences
    )

    # Decode and return stories
    stories = [tokenizer.decode(out, skip_special_tokens=True) for out in output_ids]
    return stories

if __name__ == "__main__":
    user_prompt = input("Enter the beginning of your story: ")

    # Generate 5 unique stories
    stories = generate_stories(user_prompt, num_return_sequences=5)

    print("\n--- Generated Stories ---\n")
    for i, story in enumerate(stories, 1):
        print(f"Story {i}:\n{story}\n{'-'*50}\n")
