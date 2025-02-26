import openai
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
# from openai.error import RateLimitError
openai.api_key = 'sk-5rzyydPzpzyiCAs_HQDSD7xfHBHqFn4nfgi4xn-0pYT3BlbkFJlANjKZp58WnqNuJNXQfJsQ8Ujx5cCOzKZTjjX2AjEA'  #

vowel_list = ['a', 'e', 'i', 'o', 'u']

# Load missing classes
with open('/volumes1/datasets/Imagenet/archive/Labels.json', 'r') as f:
    labels = json.load(f)

# Function to generate description using OpenAI GPT
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def generate_description_openai(class_name):
    article = 'an' if class_name[0] in vowel_list else 'a'
    prompt = f"Describe what {article} {class_name} looks like in one line. Describe its visual features so it can be identified in images."

    # try:
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.99,
        max_tokens=50,
        n=1,
        stop="."
    )
    # except RateLimitError:
    #     print("Hit rate limit. Waiting 15 seconds.")
    #     time.sleep(15)
    #     response = openai.Completion.create(
    #         model="gpt-3.5-turbo-instruct",
    #         prompt=prompt,
    #         temperature=0.99,
    #         max_tokens=50,
    #         n=1,
    #         stop="."
    #     )

    description = response["choices"][0]["text"].strip()
    return description


# Load GPT-2 model for fallback
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def generate_description_hug(class_name):
    article = 'an' if class_name[0] in vowel_list else 'a'
    prompt = f"Describe what {article} {class_name} looks like visually in one line. Describe the detailed features to help in image classification."

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.split('classification', 1)[1].strip()


# Generate descriptions for missing classes
description_dict = {}
for class_id, class_name in labels.items():
    primary_name = class_name.split(',')[0].strip()
    description = generate_description_openai(primary_name)
    description_dict[class_id] = description

# Save to JSON
output_file = '/volumes1/datasets/imagenet100_description.json'
with open(output_file, 'w') as json_file:
    json.dump(description_dict, json_file, indent=4)

print(f"Descriptions saved to {output_file}")
