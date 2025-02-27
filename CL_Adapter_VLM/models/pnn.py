# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from cl_datasets import get_dataset
from torch.optim import SGD

from utils.args import *
from utils.conf import get_device


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def get_backbone(bone, old_cols=None, x_shape=None):
    from backbone.MNISTMLP import MNISTMLP
    from backbone.MNISTMLP_PNN import MNISTMLP_PNN
    from backbone.ResNet import ResNet
    from backbone.ResNet18_PNN import resnet18_pnn

    if isinstance(bone, MNISTMLP):
        return MNISTMLP_PNN(bone.input_size, bone.output_size, old_cols)
    elif isinstance(bone, ResNet):
        return resnet18_pnn(bone.num_classes, bone.nf, old_cols, x_shape)
    else:
        raise NotImplementedError('Progressive Neural Networks is not implemented for this backbone')


class Pnn(nn.Module):
    NAME = 'pnn'
    COMPATIBILITY = ['task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Pnn, self).__init__()
        self.loss = loss
        self.args = args
        self.transform = transform
        self.device = get_device()
        self.x_shape = None
        self.nets = [get_backbone(backbone).to(self.device)]
        self.net = self.nets[-1]
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)

        self.soft = torch.nn.Softmax(dim=0)
        self.logsoft = torch.nn.LogSoftmax(dim=0)
        self.dataset = get_dataset(args)
        self.task_idx = 0

    def forward(self, x, task_label):
        if self.x_shape is None:
            self.x_shape = x.shape

        if self.task_idx == 0:
            out = self.net(x)
        else:
            self.nets[task_label].to(self.device)
            out = self.nets[task_label](x)
            if self.task_idx != task_label:
                self.nets[task_label].cpu()
        return out

    def end_task(self, dataset):
        # instantiate new column
        if self.task_idx == 4:
            return
        self.task_idx += 1
        self.nets[-1].cpu()
        self.nets.append(get_backbone(dataset.get_backbone(), self.nets, self.x_shape).to(self.device))
        self.net = self.nets[-1]
        self.opt = optim.SGD(self.net.parameters(), lr=self.args.lr)

    def observe(self, inputs, labels, not_aug_inputs):
        if self.x_shape is None:
            self.x_shape = inputs.shape

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
