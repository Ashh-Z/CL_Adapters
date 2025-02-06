import openai
import json
import os
import time
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai.error import RateLimitError

# vowel_list = ['a', 'e', 'i', 'o', 'u']
#
# # Function to generate description using OpenAI GPT
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
# def generate_description_openai(class_name):
#
#     formatted_class_name = class_name.replace('_', ' ')
#     if formatted_class_name[0] in vowel_list:
#         article = 'an'
#     else:
#         article = 'a'
#     prompt = f"Describe what {article} {formatted_class_name} looks like visually in one line. Describe the detailed features of a {class_name} to help in image classification."
#
#     try:
#         response = openai.Completion.create(
#             model="gpt-3.5-turbo-instruct",
#             prompt=prompt,
#             temperature=0.99,
#             max_tokens=50,
#             n=1,
#             stop="."
#         )
#     except RateLimitError:
#         print("Hit rate limit. Waiting 15 seconds.")
#         time.sleep(15)
#         response = openai.Completion.create(
#             model="gpt-3.5-turbo-instruct",
#             prompt=prompt,
#             temperature=0.99,
#             max_tokens=50,
#             n=1,
#             stop="."
#         )
#
#     time.sleep(0.15)
#     description = response["choices"][0]["text"].strip()
#     return description
#
#
# # Path to the DomainNet dataset
# dataset_path = '/volumes1/datasets/DN4IL/DomainNet/clipart'
#
# # Get list of class names from the folder structure
# class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
#
# # Create the JSON structure with descriptions for each class
# description_dict = {}
# for class_name in class_names:
#     print(f"Generating description for class: {class_name}")
#     description = generate_description_openai(class_name)
#     description_dict[class_name] = description


#////////////////////////////////////////////////////////////////////////////////////////////////////
GPT_PATH = '/volumes1/datasets/domainnet_description.json'
with open(GPT_PATH, 'r') as f:
    description_dict = json.load(f)

#Map to 100 classes for DN4IL
# Step 1: Extract Mapping
mapping = {}
with open('/volumes1/datasets/DN4IL/clipart_train.txt', 'r') as file:
    for line in file:
        class_name = line.split('/')[1]  # extract class name
        index = int(line.split()[-1])  # extract index
        if index not in mapping:
            mapping[index] = class_name


fin_descriptions = {}
for index, class_name in mapping.items():
    if class_name in description_dict:
        description = description_dict[class_name]
        fin_descriptions[index] = description
    else:
        fin_descriptions[index] = f"No description available for {class_name}"

# Step 3: Convert to JSON
json_descriptions = json.dumps(fin_descriptions, indent=4)
# Save JSON to a file
output_file = '/volumes1/datasets/domainnet_description_100.json'
with open(output_file, 'w') as json_file:
    json_file.write(json_descriptions)

# Save the dictionary to a JSON file


print(f"Descriptions saved to {output_file}")
