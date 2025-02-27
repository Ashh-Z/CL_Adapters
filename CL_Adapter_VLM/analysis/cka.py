import os
from PIL import Image
import torch
from torchvision import models, transforms
from torch.linalg import norm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

lst_colors = [
    # '#f9dbbd',
    "#ffffff",
    "#e1e5f2",
    '#bfdbf7',
    "#4ea8de",
    '#219ebc',
    '#022b3a',
    # '#450920',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

def hsic1(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the batched version of the Hilbert-Schmidt Independence Criterion on Gram matrices.
    This version is based on
    https://github.com/numpee/CKA.pytorch/blob/07874ec7e219ad29a29ee8d5ebdada0e1156cf9f/cka.py#L107.
    Args:
        gram_x: batch of Gram matrices of shape (bsz, n, n).
        gram_y: batch of Gram matrices of shape (bsz, n, n).
    Returns:
        a tensor with the unbiased Hilbert-Schmidt Independence Criterion values.
    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` do not have the same shape or if they do not have exactly three
        dimensions.
    """
    if len(gram_x.size()) != 3 or gram_x.size() != gram_y.size():
        raise ValueError("Invalid size for one of the two input tensors.")
    n = gram_x.shape[-1]
    gram_x = gram_x.clone()
    gram_y = gram_y.clone()
    # Fill the diagonal of each matrix with 0
    gram_x.diagonal(dim1=-1, dim2=-2).fill_(0)
    gram_y.diagonal(dim1=-1, dim2=-2).fill_(0)
    # Compute the product between k (i.e.: gram_x) and l (i.e.: gram_y)
    kl = torch.bmm(gram_x, gram_y)
    # Compute the trace (sum of the elements on the diagonal) of the previous product, i.e.: the left term
    trace_kl = kl.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
    # Compute the middle term
    middle_term = gram_x.sum((-1, -2), keepdim=True) * gram_y.sum((-1, -2), keepdim=True)
    middle_term /= (n - 1) * (n - 2)
    # Compute the right term
    right_term = kl.sum((-1, -2), keepdim=True)
    right_term *= 2 / (n - 2)
    # Put all together to compute the main term
    main_term = trace_kl + middle_term - right_term
    # Compute the hsic values
    out = main_term / (n**2 - 3 * n)
    return out.squeeze(-1).squeeze(-1)
def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.
  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)
def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.
  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.
  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]
  return gram
def cka(x, y, debiased=False):
  """Compute CKA.
  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.
  Returns:
    The value of CKA between X and Y.
  """
  x = x.type(torch.float64) if not x.dtype == torch.float64 else x
  y = y.type(torch.float64) if not y.dtype == torch.float64 else y
  # Build the Gram matrices by applying the linear kernel
  gram_x = torch.bmm(x, x.transpose(1, 2))
  gram_y = torch.bmm(y, y.transpose(1, 2))

  # Compute the HSIC values for the entire batches
  hsic1_xy = hsic1(gram_x, gram_y)
  hsic1_xx = hsic1(gram_x, gram_x)
  hsic1_yy = hsic1(gram_y, gram_y)
  # Compute the CKA value
  cka = hsic1_xy.sum() / (hsic1_xx.sum() * hsic1_yy.sum()).sqrt()
  return cka

# Load pre-trained ResNet18 model and set it to evaluation mode
model = models.resnet18(pretrained=True)
model.eval()
modules = list(model.children())[:-3]  # Remove the last few layers to extract intermediate features
resnet18_features = nn.Sequential(*modules)

# Define image transformation: resize, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Function to load and process image
def load_and_process_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert image to RGB
    return transform(image).unsqueeze(0)  # Add batch dimension

image_folder = "/volumes1/datasets/DN4IL/tmp"
image_files = []
domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
for dom in domains:
    image_files.append(os.path.join(image_folder, '{}_002_000002.jpg'.format(dom)))
images = [load_and_process_image(path) for path in image_files]

# Extract features and flatten them
features = []
for image in images:
    with torch.no_grad():
        feature = resnet18_features(image)  # Extract intermediate features
        feature = feature.view(feature.size(0), feature.size(1), -1) #feature.view(feature.size(0), -1)  # Flatten (batch_size, channels * height * width)
        feature = feature.mean(dim=2, keepdim=True)
        features.append(feature)  # Remove the batch dimensiona

# Compute pairwise CKA similarities (6x6 matrix
similarity_matrix = torch.zeros((6, 6))
for i in range(6):
    for j in range(6):
        similarity_matrix[i, j] = cka(features[i], features[j])


masked_similarity_matrix = np.tril(similarity_matrix.numpy())
print(masked_similarity_matrix)# Convert to NumPy array and get lower triangle
# Plot the CKA similarity matrix
fig, ax = plt.subplots()
plt.imshow(masked_similarity_matrix, cmap=custom1, interpolation='nearest')
plt.colorbar()
plt.title("CKA Similarity Matrix")
plt.xticks(range(6), domains, rotation=45)
plt.yticks(range(6), domains, rotation=45)
# plt.savefig("/volumes1/vlm-cl/paper/cka2.png", bbox_inches='tight')
plt.show()
