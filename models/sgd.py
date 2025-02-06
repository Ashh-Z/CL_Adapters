# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# from models.utils.continual_model import ContinualModel
# from utils.args import *


# def get_parser() -> ArgumentParser:
#     parser = ArgumentParser(description='Continual Learning via'
#                                         ' Progressive Neural Networks.')
#     add_management_args(parser)
#     add_experiment_args(parser)
#     return parser


# class Sgd(ContinualModel):
#     NAME = 'sgd'
#     COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

#     def __init__(self, backbone, loss, args, transform):
#         super(Sgd, self).__init__(backbone, loss, args, transform)

#     # def observe(self, inputs, labels, not_aug_inputs):
#     #     self.opt.zero_grad()
#     #     outputs = self.net(inputs)
#     #     loss = self.loss(outputs, labels)
#     #     loss.backward()
#     #     self.opt.step()

#     #     return loss.item()

#     def observe(self, inputs, labels, not_aug_inputs, dataset=None):
#         print(f' ****** inside SGD observe ****** ')
#         self.opt.zero_grad()
#         outputs = self.net(inputs)
#         loss = self.loss(outputs, labels)
#         loss.backward()
#         self.opt.step()

#         return loss.item()

from models.utils.continual_model import ContinualModel
from utils.args import *
import os
import torch


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                     ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Sgd(ContinualModel):
    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.bn_frozen = False  # flag to ensure BN layers are frozen only once

    def freeze_bn_layers(self):
        """
        Freeze all BatchNorm layers in the network.
        The BN layers are set to evaluation mode and their parameters are no longer updated.
        """
        for module in self.net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()  # Prevent BN layers from updating running statistics
                for param in module.parameters():
                    param.requires_grad = False
        print("BatchNorm layers have been frozen.")

    def begin_task(self, dataset):
        """
        Prepare for training on a new task.
        Freeze the BN layers before starting any tasks (only once).
        """
        if not self.bn_frozen:
            self.freeze_bn_layers()
            self.bn_frozen = True
        self.dataset = dataset  # Store dataset for label handling

    def observe(self, inputs, labels, not_aug_inputs, task_id=None):
        """
        Observation method for training the model.
        :param inputs: training inputs
        :param labels: training labels (global labels)
        :param not_aug_inputs: non-augmented inputs
        :param task_id: task identifier (needed for multihead)
        :return: loss value
        """
        print(f' ****** inside SGD observe ****** ')
        self.opt.zero_grad()
        
        # Handle multihead case
        if hasattr(self.net, 'multihead') and self.net.multihead:
            if task_id is None:
                raise ValueError("Task ID must be provided for multihead training")
                
            print(f' ***** INSIDE SGD observe labels (global): {labels} ***** ')
            print(f' ***** INSIDE SGD observe task_id: {task_id} ***** ')
            outputs = self.net(inputs, task_id=task_id)
            
            # Adjust the global labels to task-local.
            # For zero-indexed global labels, if task0 labels are 0-19 and task1 labels are 20-39, etc.
            if hasattr(self, 'dataset') and hasattr(self.dataset, 'N_CLASSES_PER_TASK'):
                # For a proper conversion, subtract task_id * N_CLASSES_PER_TASK.
                offset = int(task_id) * self.dataset.N_CLASSES_PER_TASK
                print(f"Adjusting labels with offset: {offset}")
                labels = labels - offset
        else:
            outputs = self.net(inputs)
            
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()

    def end_task(self, dataset):
        """
        Saves the model after each task. 
        Specifically saves:
          1. Complete model state.
          2. Task-specific adapter weights if using adapter-based training.
          3. Freezes the task-specific classifier so that it remains unchanged in future tasks.
        """
        self.save_models(dataset)
        if hasattr(self.args, 'use_adapter') and self.args.use_adapter:
            adapter_dir = os.path.join(
                self.args.output_dir, 
                "adapters",
                dataset.SETTING, 
                dataset.NAME, 
                self.NAME, 
                self.args.experiment_id
            )
            os.makedirs(adapter_dir, exist_ok=True)
            if hasattr(self.net, 'get_adapter_weights'):
                adapter_state = self.net.get_adapter_weights(str(self.current_task))
                if adapter_state is not None:
                    adapter_path = os.path.join(adapter_dir, f'adapter_task{self.current_task}.pt')
                    torch.save(adapter_state, adapter_path)
                    print(f"Saved adapter weights for task {self.current_task} to {adapter_path}")

        # Freeze the classifier head for the current (completed) task in multihead mode.
        if hasattr(self.net, 'multihead') and self.net.multihead:
            task_id = str(self.current_task)
            if task_id in self.net.classifiers:
                for param in self.net.classifiers[task_id].parameters():
                    param.requires_grad = False
                print(f"Classifier head for task {task_id} has been frozen.")

        # Optionally: Freeze the BN layers at the end of each task in the backbone to avoid running statistics update.
        # Example: iterate over all submodules and set BN layers to eval mode.
        # for m in self.net.modules():
        #     if isinstance(m, torch.nn.BatchNorm2d):
        #         m.eval()

        self.current_task += 1