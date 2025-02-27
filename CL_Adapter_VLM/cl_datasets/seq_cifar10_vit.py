# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import clip
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet import *
from backbone.ResNet_mam import *
from backbone.vit import vittiny
from backbone.vit_llm import vittinyllm
# from backbone.ResNet_mam_llm import *
from PIL import Image
from torchvision.datasets import CIFAR10

import os
from cl_datasets.transforms.denormalization import DeNormalize
from cl_datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from cl_datasets.utils.validation import get_train_val

class TCIFAR10(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())
        print(f' ===== seq_cifar10_vit ===== ')

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCIFAR10Vit(ContinualDataset):

    NAME = 'seq-cifar10-vit'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    CLASS_ID = {0: "car (automobile)", 1: "airplane", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                6: "frog", 7: "horse", 8: "cargo ship", 9: "truck"}

    print(f' ===== seq_cifar10_vit ===== ')

    #for CLIP model
    # TRANSFORM = transforms.Compose(
    #         [transforms.Resize((224, 224)),
    #          transforms.RandomHorizontalFlip(),
    #          transforms.ToTensor(),
    #          transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                               (0.2470, 0.2435, 0.2615))])
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(os.path.join(self.args.dataset_dir, 'CIFAR10'), train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR10(os.path.join(self.args.dataset_dir, 'CIFAR10'), train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10Vit.TRANSFORM])
        return transform

    def get_backbone(self):
        if self.args.arch == 'clip_vit':
            from transformers import CLIPModel, CLIPProcessor
            # model, _ = clip.load("ViT-B/32", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            return model.vision_model
        elif self.args.arch == 'clip_res':
            # Load the CLIP model and extract the vision encoder
            model, _ = clip.load("RN50", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            return model.visual  # Return the vision encoder part of CLIP
        elif self.args.arch == 'vittiny':
            model = vittiny(SequentialCIFAR10Vit.N_CLASSES_PER_TASK * SequentialCIFAR10Vit.N_TASKS)
            return model
        elif self.args.arch == "vittinyllm":
            model = vittinyllm(SequentialCIFAR10Vit.N_CLASSES_PER_TASK * SequentialCIFAR10Vit.N_TASKS, self.args.llm_block)
            return model
        else:
            raise (RuntimeError("architecture type not found"))

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR10.get_batch_size()
