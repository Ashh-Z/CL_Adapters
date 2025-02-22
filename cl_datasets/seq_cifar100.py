# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet import *
from backbone.ResNet_mam import *
from backbone.ResNet_mam_llm import *
from PIL import Image
from torchvision.datasets import CIFAR100

from cl_datasets.transforms.denormalization import DeNormalize
from cl_datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from cl_datasets.utils.validation import get_train_val
# from utils.conf import base_path_dataset as base_path
import os
from argparse import Namespace


class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

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


class SequentialCIFAR100(ContinualDataset):

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])
    # CLASS_ID = {}
    # with open('cl_datasets/metadata/cifar100_class_mapping.json', 'r') as file:
    #     CLASS_ID = json.load(file)

    def __init__(self, args: Namespace) -> None:
        super(SequentialCIFAR100, self).__init__(args)

        num_cls = 100 #SequentialCIFAR100.N_TASKS * SequentialCIFAR100.N_CLASSES_PER_TASK
        SequentialCIFAR100.N_TASKS = args.n_tasks_cif
        SequentialCIFAR100.N_CLASSES_PER_TASK = num_cls // args.n_tasks_cif
        assert SequentialCIFAR100.N_TASKS * SequentialCIFAR100.N_CLASSES_PER_TASK == num_cls, \
            "n_tasks_cif should be a divisor of number of classes of CIFAR-100 i.e 100"
        print(f'Setting Number of tasks to {SequentialCIFAR100.N_TASKS} with {SequentialCIFAR100.N_CLASSES_PER_TASK} classes each')

    def get_examples_number(self):
        train_dataset = MyCIFAR100(os.path.join(self.args.dataset_dir, 'CIFAR100'), train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR100(os.path.join(self.args.dataset_dir, 'CIFAR100'), train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(os.path.join(self.args.dataset_dir, 'CIFAR100'), train=False,
                                   download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    def get_backbone(self):
        if self.args.arch == 'resnet18':
            return resnet18(SequentialCIFAR100.N_CLASSES_PER_TASK
                        * SequentialCIFAR100.N_TASKS)
        elif self.args.arch == 'resnet18mam':
            return resnet18mam(SequentialCIFAR100.N_CLASSES_PER_TASK
                           * SequentialCIFAR100.N_TASKS)
        elif self.args.arch == 'resnet50mam':
            return resnet50mam(SequentialCIFAR100.N_CLASSES_PER_TASK
                       * SequentialCIFAR100.N_TASKS)
        elif self.args.arch == 'resnet18mamllm':
            return resnet18mamllm(SequentialCIFAR100.N_CLASSES_PER_TASK
                          * SequentialCIFAR100.N_TASKS, 64, self.args.llm_block)
        else:
            raise (RuntimeError("architecture type not found"))

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        # return 50
        return 2

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR100.get_batch_size()

    @staticmethod
    def get_scheduler(model, args):
        return None

    # @staticmethod
    # def get_scheduler(model, args) -> torch.optim.lr_scheduler:
    #     model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
    #     return scheduler

