import os
import  json
# Load Tiny ImageNet WNIDs
tiny_imagenet_wnids = []
with open('/volumes1/datasets/tiny-imagenet-200/wnids.txt', 'r') as f:
    tiny_imagenet_wnids = [line.strip() for line in f.readlines()]

# Load ImageNet-R WNIDs
imagenet_r_wnids = []
with open('/volumes1/datasets/imagenet-r/README.txt', 'r') as f:
    for line in f:
        if line.startswith('n'):
            imagenet_r_wnids.append(line.split()[0].strip())

# Create a mapping of WNIDs from Tiny ImageNet to ImageNet-R
wnid_to_idx = {wnid: idx for idx, wnid in enumerate(tiny_imagenet_wnids)}
overlap_wnids = set(tiny_imagenet_wnids).intersection(set(imagenet_r_wnids))

# Create a final mapping to be used for label alignment
final_mapping = {wnid: wnid_to_idx[wnid] for wnid in overlap_wnids}
# Save the final mapping using json
mapping_file = '/volumes1/datasets/imagenet-r/final_mapping.json'
with open(mapping_file, 'w') as f:
    json.dump(final_mapping, f)

print(f"Mapping saved to {mapping_file}")