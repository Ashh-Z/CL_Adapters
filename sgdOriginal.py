# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
from utils.args import *


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

    def observe(self, inputs, labels, not_aug_inputs, dataset=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()


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

# from models.utils.continual_model import ContinualModel
# from utils.args import *
# import os
# import torch


# def get_parser() -> ArgumentParser:
#     parser = ArgumentParser(description='Continual Learning via'
#                                      ' Progressive Neural Networks.')
#     add_management_args(parser)
#     add_experiment_args(parser)
#     return parser


# class Sgd(ContinualModel):
#     NAME = 'sgd'
#     COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

#     def __init__(self, backbone, loss, args, transform):
#         super(Sgd, self).__init__(backbone, loss, args, transform)
#         self.current_task = 0

#     def observe(self, inputs, labels, not_aug_inputs, task_id=None):
#         """
#         Observation method for training the model.
#         :param inputs: training inputs
#         :param labels: training labels
#         :param not_aug_inputs: non-augmented inputs
#         :param task_id: task identifier (needed for multihead)
#         :return: loss value
#         """
#         print(f' ****** inside SGD observe ****** ')
#         self.opt.zero_grad()
        
#         # Handle multihead case
#         if hasattr(self.net, 'multihead') and self.net.multihead:
#             if task_id is None:
#                 raise ValueError("Task ID must be provided for multihead training")
                
#             # Adjust labels for the current task
#             if hasattr(self, 'dataset'):
#                 # Adjust labels to be within range [0, N_CLASSES_PER_TASK)
#                 labels = labels % self.dataset.N_CLASSES_PER_TASK
                
#             outputs = self.net(inputs, task_id=task_id)
#         else:
#             outputs = self.net(inputs)
            
#         # Print shape information for debugging
#         print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
#         print(f"Labels min: {labels.min()}, Labels max: {labels.max()}")
#         print(f"Outputs size (n_classes): {outputs.size(1)}")
        
#         loss = self.loss(outputs, labels)
#         loss.backward()
#         self.opt.step()

#         return loss.item()

#     def end_task(self, dataset):
#         """
#         Saves the model after each task. 
#         Specifically saves:
#         1. Complete model state
#         2. Task-specific adapter weights if using adapter-based training
#         """
#         # Save complete model state using parent class method
#         self.save_models(dataset)
        
#         # Additional saving for adapter-based model
#         if hasattr(self.args, 'use_adapter') and self.args.use_adapter:
#             # Create adapter-specific save directory
#             adapter_dir = os.path.join(
#                 self.args.output_dir, 
#                 "adapters",
#                 dataset.SETTING, 
#                 dataset.NAME, 
#                 self.NAME, 
#                 self.args.experiment_id
#             )
#             os.makedirs(adapter_dir, exist_ok=True)
            
#             # Save task-specific adapter weights
#             if hasattr(self.net, 'get_adapter_weights'):
#                 adapter_state = self.net.get_adapter_weights(str(self.current_task))
#                 if adapter_state is not None:
#                     adapter_path = os.path.join(adapter_dir, f'adapter_task{self.current_task}.pt')
#                     torch.save(adapter_state, adapter_path)
#                     print(f"Saved adapter weights for task {self.current_task} to {adapter_path}")
            
#         self.current_task += 1

#     def begin_task(self, dataset):
#         """
#         Prepare for training on a new task
#         """
#         self.dataset = dataset  # Store dataset for label handling