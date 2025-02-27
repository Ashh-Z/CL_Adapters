# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.aux_utils import AuxiliaryNet
from utils.vision_lang import lossVLM

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning Through'
                                        ' Synaptic Intelligence.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_auxiliary_args(parser)
    parser.add_argument('--c', type=float, required=True,
                        help='surrogate loss weight parameter c')
    parser.add_argument('--xi', type=float, required=True,
                        help='xi parameter for EWC online')

    return parser


class VLSI(ContinualModel):
    NAME = 'vl_si'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(VLSI, self).__init__(backbone, loss, args, transform)

        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0
        self.aux = AuxiliaryNet(self.args, self.device)
        self.kd_loss = lossVLM(self)

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        # big omega calculation step
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.net.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.net.get_params().data - self.checkpoint) ** 2 + self.args.xi)

        # store parameters checkpoint and reset small_omega
        self.checkpoint = self.net.get_params().data.clone().to(self.device)
        self.small_omega = 0

    def observe(self, inputs, labels, not_aug_inputs, dataset=None):
        loss = 0
        self.opt.zero_grad()

        outputs, features = self.net(inputs, returnt='all')
        loss_aux = self.kd_loss.loss_vlm(labels, dataset, features)

        penalty = self.penalty()
        loss_ce1 = self.loss(outputs, labels) + self.args.c * penalty
        loss = loss_ce1 + loss_aux

        loss.backward()
        nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
        self.opt.step()

        self.small_omega += self.args.lr * self.net.get_grads().data ** 2

        return loss.item()
