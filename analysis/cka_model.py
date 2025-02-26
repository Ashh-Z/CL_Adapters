import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.linalg import norm
from matplotlib.colors import LinearSegmentedColormap
from backbone.ResNet_mam import resnet18mam  # Import your custom ResNet implementation
from backbone.ResNet_mam_llm import resnet18mamllm
# Define a custom colormap
lst_colors = ["#ffffff", "#e1e5f2", '#bfdbf7', "#4ea8de", '#219ebc', '#022b3a']
custom1 = LinearSegmentedColormap.from_list(name='custom', colors=lst_colors)

# Define image transformation: resize, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess images
def load_and_process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def generate_table_markdown(matrix, domains):
    markdown = "| Domain     | " + " | ".join(domains) + " |\n"
    markdown += "|-" + "-|" * len(domains) + "\n"
    for i, row in enumerate(matrix):
        row_values = " | ".join(f"{val:.2f}" for val in row)
        markdown += f"| {domains[i]} | {row_values} |\n"
    return markdown
# Define CKA computation
class LinearCKA:
    def __init__(self, device='cuda'):
        self.device = device

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def centering(self, K):
        n = K.shape[0]
        H = self.get_centering_matrix(n)
        return torch.matmul(torch.matmul(H, K), H)

    def get_centering_matrix(self, n):
        unit = torch.ones(n, n).to(self.device)
        I = torch.eye(n).to(self.device)
        return I - unit / n

    def calculate(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        return hsic / (var1 * var2)

# Load the custom ResNet model
def load_resnet_model(checkpoint_path, nclasses=100):
    if "ix" in checkpoint_path:
        model = resnet18mamllm(nclasses=nclasses, llm_block='sent_transf').cuda() # Use your custom ResNet18
    else:
        model = resnet18mam(nclasses=nclasses).cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'], strict=True)  # Adjust based on checkpoint structure
    model.eval()  # Set the model to evaluation mode
    return model

# Define paths and domains
image_folder = "/volumes1/datasets/DN4IL/tmp/ice"
# checkpoint_path = "/volumes1/vlm-cl/rebuttal/dn4il/vl_er/exsave-resnet18mam-dn4il-b200-lr-0.03-e50-8.0-t-sent_transf-s-5/model_task6.ph"
# checkpoint_path="/volumes1/vlm-cl/rebuttal/results/domain-il/dn4il/er/model-er-l0.1-resnet18mam-dn4il-buf-200-s-2/model_task6.ph"  # Base
checkpoint_path = '/volumes1/vlm-cl/rebuttal/dn4il/er/ixsave-resnet18mamllm-dn4il-b200-lr-0.0001-wd-0.01-sent_transf-s-3/model_task6.ph'
domains = ['real', 'clipart', 'infograph', 'painting', 'sketch', 'quickdraw']

# Collect domain-specific image paths
all_files = os.listdir(image_folder)
image_files = []
for dom in domains:
    domain_files = [os.path.join(image_folder, f) for f in all_files if f.startswith(dom) and f.endswith('.jpg')]
    image_files.extend(domain_files)

# Load images and preprocess
images = [load_and_process_image(path) for path in image_files]

# Extract features using the loaded ResNet model
model = load_resnet_model(checkpoint_path, nclasses=100)  # Adjust nclasses as needed
features = []
for image in images:
    with torch.no_grad():
        feature = model(image.cuda(), returnt='features').cuda()  # Adjust for your ResNet function
        # feature = feature.unsqueeze(-1)  #feature.mean(dim=2).permute(1, 0)  # Reduce spatial dimensions and permute
        feature = feature.view(feature.size(0), feature.size(1), -1) #feature.view(feature.size(0), -1)  # Flatten (batch_size, channels * height * width)
        feature = feature.mean(dim=2).permute(1,0)
        features.append(feature)

# Compute pairwise CKA similarities
cka = LinearCKA(device='cuda')
similarity_matrix = torch.zeros((len(domains), len(domains)))
for i in range(len(domains)):
    for j in range(len(domains)):
        similarity_matrix[i, j] = cka.calculate(features[i], features[j])

# Mask and visualize the similarity matrix
masked_similarity_matrix = np.tril(similarity_matrix.numpy())  # Convert to NumPy and mask
fig, ax = plt.subplots()
plt.imshow(masked_similarity_matrix, cmap=custom1, interpolation='nearest')
plt.colorbar()
plt.title("CKA Similarity Matrix")
plt.xticks(range(len(domains)), domains, rotation=45)
plt.yticks(range(len(domains)), domains, rotation=45)
# plt.savefig("/volumes1/vlm-cl/paper/cka_custom_resnet.png", bbox_inches='tight')
plt.show()

markdown_table = generate_table_markdown(masked_similarity_matrix, domains)
print(markdown_table)