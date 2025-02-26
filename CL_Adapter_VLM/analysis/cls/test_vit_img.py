import os
import timm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a pretrained ViT model
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=100)
model = model.to(device)
model.eval()

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet validation dataset
val_dataset = datasets.ImageFolder(os.path.join("/volumes1/datasets/Imagenet/archive", 'val.X'), transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

# Evaluate accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Baseline Top-1 Accuracy: {accuracy:.2f}%")
