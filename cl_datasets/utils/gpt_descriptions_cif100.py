import openai
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
from openai.error import RateLimitError

# Add your OpenAI API key here
openai.api_key = 'sk-5rzyydPzpzyiCAs_HQDSD7xfHBHqFn4nfgi4xn-0pYT3BlbkFJlANjKZp58WnqNuJNXQfJsQ8Ujx5cCOzKZTjjX2AjEA'  #
vowel_list = ['a', 'e', 'i', 'o', 'u']

# Function to generate description using OpenAI GPT
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def generate_description_openai(class_name):
    if class_name[0] in vowel_list:
        article = 'an'
    else:
        article = 'a'
    prompt = "Describe what " + article + " " + class_name + " looks like in one line. Describe its visual feature so that it can be identified in images." #f"Generate two one-line descriptions for the class '{class_name}':"
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo", #"gpt-3.5-turbo-instruct"
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     # prompt=prompt,
    #     max_tokens=50,
    #     n=1,
    #     stop=None,
    #     temperature=0.7
    # )
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=0.99,
            max_tokens=50,
            n=1,
            stop="."
        )
    except RateLimitError:
        print("Hit rate limit. Waiting 15 seconds.")
        time.sleep(15)
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=.99,
            max_tokens=50,
            n=1,
            stop="."
        )
    time.sleep(0.15)
    description = response["choices"][0]["text"] #response['choices'][0]['message']['content'].strip()
    print(description)
    return description

# Load pre-trained model and tokenizer from Hugging Face
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# Function to generate description using GPT-2
def generate_description_hug(class_name):
    if class_name[0] in vowel_list:
        article = 'an'
    else:
        article = 'a'
    prompt = "Describe what " + article + " " + class_name + " looks like visually in one line. Describe the detailed features of a " + class_name + " to help in image classification"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    description = description.split('classification', 1)[1].strip()
    return description

# Open and read the JSON file
with open('/volumes1/datasets/cifar/CIFAR100/cifar100_class_mapping.json', 'r') as file:
    class_dict = json.load(file)

# Create the JSON structure with descriptions for each class ID
description_dict = {}
for class_id, class_name in class_dict.items():
        # Split class names by comma and generate descriptions for each
        synonyms = [name.strip() for name in class_name.split(',')]
        descriptions = generate_description_openai(synonyms[0]) #[generate_description(name) for name in synonyms]
        description_dict[class_id] = descriptions

# Save the dictionary to a JSON file
output_file = '/volumes1/datasets/cifar100_description.json'
with open(output_file, 'w') as json_file:
    json.dump(description_dict, json_file, indent=4)

print(f"Descriptions saved to {output_file}")
