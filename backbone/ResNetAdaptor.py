# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini,
# Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu

# If using torchvision's pre-trained models:
import torchvision

from backbone import MammothBackbone


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor
        :return: output tensor
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: nn.Module, num_blocks: List[int],
                 num_classes: int, nf: int,
                 pretrained: bool = False,
                 pretrained_path: Optional[str] = None,
                 multihead: bool = False,
                 freeze_backbone: bool = False) -> None:
        """
        Instantiates the network.
        :param block: the basic building block (e.g., BasicBlock or Bottleneck)
        :param num_blocks: list with the number of blocks per layer.
        :param num_classes: default number of output classes (for task-specific head).
        :param nf: number of filters.
        :param pretrained: whether to load pre-trained weights.
        :param pretrained_path: optional path for pre-trained weights.
        :param multihead: if True, a ModuleDict of task-specific classifier heads is created.
        :param freeze_backbone: if True, all backbone layers (except the classifier head(s)) are frozen.
        """
        super(ResNet, self).__init__()
        self.block = block
        self.nf = nf
        self.in_planes = nf
        self.multihead = multihead

        # Backbone layers.
        self.conv1 = nn.Conv2d(3, nf, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(nf)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # Global pooling: output dimension.
        backbone_out_dim = nf * 8 * block.expansion

        # Default classifier (if not in multihead mode)
        self.classifier = nn.Linear(backbone_out_dim, num_classes)

        # For multihead mode, keep a dictionary of classifiers.
        if self.multihead:
            self.classifiers = nn.ModuleDict()

        # Optionally load pre-trained weights.
        if pretrained:
            self.load_pretrained_weights(pretrained_path)

        # Freeze the entire backbone except for the classifier/classifiers.
        if freeze_backbone:
            for name, param in self.named_parameters():
                # Do not freeze parameters from the task-specific classifiers.
                if not (name.startswith("classifier") or name.startswith("classifiers")):
                    param.requires_grad = False
            print("Backbone parameters frozen. Only classifier layers will be trained.")

    def set_adapter_weights(self, task_id: str, adapter_state: dict) -> None:
        """
        Sets the adapter weights for a specific task.
        
        :param task_id: task identifier as a string
        :param adapter_state: dictionary containing saved adapter weights
        """
        if not self.multihead:
            raise ValueError("set_adapter_weights only works in multihead mode")
            
        # If task_id is not in classifiers, add it
        if task_id not in self.classifiers and 'classifier.weight' in adapter_state:
            # Extract number of classes from the adapter weights
            num_classes = adapter_state['classifier.weight'].size(0)
            self.add_task_classifier(task_id, num_classes)
            
        # Load classifier weights for the task
        if task_id in self.classifiers:
            if f'classifiers.{task_id}.weight' in adapter_state:
                self.classifiers[task_id].weight.data = adapter_state[f'classifiers.{task_id}.weight']
                self.classifiers[task_id].bias.data = adapter_state[f'classifiers.{task_id}.bias']
            elif 'classifier.weight' in adapter_state:
                # Compatibility with olf--der saves that might use a different key format
                self.classifiers[task_id].weight.data = adapter_state['classifier.weight']
                self.classifiers[task_id].bias.data = adapter_state['classifier.bias']
                
        print(f"Set adapter weights for task {task_id}")
        
    def get_adapter_weights(self, task_id: str) -> dict:
        """
        Gets the adapter weights for a specific task.
        
        :param task_id: task identifier as a string
        :return: dictionary containing adapter weights, or None if task_id not found
        """
        if not self.multihead:
            raise ValueError("get_adapter_weights only works in multihead mode")
            
        if task_id not in self.classifiers:
            print(f"Warning: Task {task_id} not found in classifiers")
            return None
            
        adapter_state = {}
        # Save classifier weights
        adapter_state[f'classifiers.{task_id}.weight'] = self.classifiers[task_id].weight.data
        adapter_state[f'classifiers.{task_id}.bias'] = self.classifiers[task_id].bias.data
        
        # For compatibility with older code, also save under generic keys
        adapter_state['classifier.weight'] = self.classifiers[task_id].weight.data
        adapter_state['classifier.bias'] = self.classifiers[task_id].bias.data
        
        return adapter_state

    def _make_layer(self, block: nn.Module, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def load_pretrained_weights(self, pretrained_path=None):
        """
        Load pretrained weights from path or torchvision's default if path is None
        """
        print("Loading pre-trained weights for the backbone...")
        
        # Change this line to check block type without instantiation
        is_resnet18_or_34 = self.block == BasicBlock  # Compare the class directly
        
        if pretrained_path:
            # Load from provided path
            state_dict = torch.load(pretrained_path)
            self.load_state_dict(state_dict)
        else:
            # Load from torchvision's pre-trained models
            if is_resnet18_or_34:
                # ResNet18 or ResNet34
                pretrained_model = torchvision.models.resnet18(pretrained=True)
            else:
                # ResNet50 or deeper
                pretrained_model = torchvision.models.resnet50(pretrained=True)
                
            # Copy weights for shared layers
            own_state = self.state_dict()
            for name, param in pretrained_model.state_dict().items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)

    def add_task_classifier(self, task_id: str, num_classes: int):
        """
        Adds a new classifier head for the given task.
        :param task_id: a unique string identifier for the task.
        :param num_classes: number of classes for the task.
        """
        if not self.multihead:
            raise ValueError("Multihead is not enabled for this model.")
        backbone_out_dim = self.nf * 8 * self.block.expansion
        self.classifiers[task_id] = nn.Linear(backbone_out_dim, num_classes)
        print(f"Added classifier head for task {task_id} with {num_classes} classes.")

    def forward(self, x: torch.Tensor, returnt: str = 'out',
                task_id: Optional[str] = None) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :param task_id: (optional) task identifier used in multihead mode.
                        If multihead is enabled and task_id is not provided or not found,
                        an error is raised.
        :return: output tensor (output_classes) or a tuple of (output, features)
        """
        out = relu(self.bn1(self.conv1(x)))  # e.g., (64, H/2, W/2)
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out = self.layer1(out)  # e.g., (64, H/4, W/4)
        out = self.layer2(out)  # e.g., (128, H/8, W/8)
        out = self.layer3(out)  # e.g., (256, H/16, W/16)
        out = self.layer4(out)  # e.g., (512, H/32, W/32)
        out = avg_pool2d(out, out.shape[2])  # Global average pooling -> (B, channels, 1, 1)
        feature = out.view(out.size(0), -1)  # Flatten to (B, channels)

        if returnt == 'features':
            # print(f' ===== INSIDE ResNetAdaptor forward features: {feature} ===== ')
            return feature

        # If multihead is enabled, use the classifier corresponding to the task_id.
        if self.multihead:
            if task_id is None or task_id not in self.classifiers:
                raise ValueError("In multihead mode a valid task_id must be provided.")
            # print(f' ===== INSIDE ResNetAdaptor forward task_id: {task_id} ===== ')
            out = self.classifiers[task_id](feature)
        else:
            # print(f' ===== INSIDE ResNetAdaptor forward classifier NO TASK ID !!!!! : {self.classifier} ===== ')
            out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feature)

        raise NotImplementedError("Unknown return type")


def resnet18(nclasses: int, nf: int = 64,
             pretrained: bool = True,
             pretrained_path: Optional[str] = None,
             multihead: bool = True,
             freeze_backbone: bool = False) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes for the default head.
    :param nf: number of filters.
    :param pretrained: whether to load pre-trained weights.
    :param pretrained_path: optional path for pre-trained weights.
    :param multihead: whether to enable multihead (task incremental) mode.
    :param freeze_backbone: if True, backbone layers are frozen.
    :return: ResNet network.
    """
    print(' ****** Creating RESNET18 ******  ')
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf,
                  pretrained=pretrained,
                  pretrained_path=pretrained_path,
                  multihead=multihead,
                  freeze_backbone=freeze_backbone)


def resnet50(nclasses: int, nf: int = 64,
             pretrained: bool = True,
             pretrained_path: Optional[str] = None,
             multihead: bool = True,
             freeze_backbone: bool = False) -> ResNet:
    """
    Instantiates a ResNet50 network.
    :param nclasses: number of output classes for the default head.
    :param nf: number of filters.
    :param pretrained: whether to load pre-trained weights.
    :param pretrained_path: optional path for pre-trained weights.
    :param multihead: whether to enable multihead (task incremental) mode.
    :param freeze_backbone: if True, backbone layers are frozen.
    :return: ResNet network.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf,
                  pretrained=pretrained,
                  pretrained_path=pretrained_path,
                  multihead=multihead,
                  freeze_backbone=freeze_backbone)
