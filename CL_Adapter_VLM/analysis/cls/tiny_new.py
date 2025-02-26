import os
import shutil

# Load Tiny ImageNet WNIDs
tiny_imagenet_wnids = []
with open('/volumes1/datasets/tiny-imagenet-200/wnids.txt', 'r') as f:
    tiny_imagenet_wnids = [line.strip() for line in f.readlines()]

# Load ImageNet-R WNIDs
imagenet_r_wnids = []
with open('/volumes1/datasets/imagenet-o/README.txt', 'r') as f:
    for line in f:
        if line.startswith('n'):
            imagenet_r_wnids.append(line.split()[0].strip())

# Find overlapping WNIDs
overlapping_wnids = set(tiny_imagenet_wnids).intersection(set(imagenet_r_wnids))
print(f"Found {len(overlapping_wnids)} overlapping classes.")

# Path to the ImageNet-R dataset
imagenet_r_path = '/volumes1/datasets/imagenet-o'
new_dataset_path = '/volumes1/datasets/imagenet-o-map'

# Create new dataset folder for overlapping classes
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# Copy only the overlapping class images
for wnid in overlapping_wnids:
    class_folder = os.path.join(imagenet_r_path, wnid)
    if os.path.exists(class_folder):
        shutil.copytree(class_folder, os.path.join(new_dataset_path, wnid))
    else:
        print(f"Class folder for {wnid} not found in ImageNet-R.")

print(f"Dataset with overlapping classes created at {new_dataset_path}")
