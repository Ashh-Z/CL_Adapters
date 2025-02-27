# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from cl_datasets import get_dataset
from cl_datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.tb_logger import *
from utils.loggers import *
from utils.status import ProgressBar
import numpy as np

# try:
#     import wandb
#     wandb.login(key='fa9d5ad248f922603618680d1197fcb953d7d32e')
# except ImportError or AttributeError:
#     wandb = None

wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')
    
    # print(f' ***** INSIDE mask_classes outputs: {outputs} ***** ')
    # print(f' ***** INSIDE mask_classes dataset: {dataset} ***** ')
    # print(f' ***** INSIDE mask_classes k: {k} ***** ')


# def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
#     """
#     Evaluates the accuracy of the model for each past task.
#     :param model: the model to be evaluated
#     :param dataset: the continual dataset at hand
#     :return: a tuple of lists, containing the class-il
#              and task-il accuracy for each task
#     """
#     status = model.net.training
#     model.net.eval()
#     accs, accs_mask_classes = [], []
#     for k, test_loader in enumerate(dataset.test_loaders):
#         if last and k < len(dataset.test_loaders) - 1:
#             continue
#         correct, correct_mask_classes, total = 0.0, 0.0, 0.0
#         for data in test_loader:
#             with torch.no_grad():
#                 inputs, labels = data
#                 inputs, labels = inputs.to(model.device), labels.to(model.device)

#                 if 'class-il' not in model.COMPATIBILITY:
#                     outputs = model(inputs, k)
#                 else:
#                     outputs = model(inputs)

#                 if isinstance(outputs, tuple):
#                     outputs = outputs[0]

#                 _, pred = torch.max(outputs.data, 1)
#                 correct += torch.sum(pred == labels).item()
#                 total += labels.shape[0]

#                 if dataset.SETTING == 'class-il':
#                     mask_classes(outputs, dataset, k)
#                     _, pred = torch.max(outputs.data, 1)
#                     correct_mask_classes += torch.sum(pred == labels).item()

#         print(f'Task {k} Accuracy: {correct / total * 100}')  # Added for eval
#         accs.append(correct / total * 100
#                     if 'class-il' in model.COMPATIBILITY else 0)
#         accs_mask_classes.append(correct_mask_classes / total * 100)

#     model.net.train(status)
#     return accs, accs_mask_classes
# ----------------------------------------------------------------------------------------------
# def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
#     """
#     Evaluates the accuracy of the model for each past task.
#     Adjusts the global labels to local ones for multihead networks.
#     """
#     print(f' ***** STARTING evaluate ****** ')
#     status = model.net.training
#     model.net.eval()
#     accs, accs_mask_classes = [], []
    
#     for k, test_loader in enumerate(dataset.test_loaders):
#         if last and k < len(dataset.test_loaders) - 1:
#             continue
            
#         # Loop over the test set for task k
#         correct, correct_mask_classes, total = 0.0, 0.0, 0.0
#         for data in test_loader:
#             with torch.no_grad():
#                 inputs, labels = data
#                 # print(f' ----- labels (global): {labels} ----- ')
#                 inputs, labels = inputs.to(model.device), labels.to(model.device)

#                 # For multihead, pass the correct task_id for evaluation.
#                 if hasattr(model.net, 'multihead') and model.net.multihead:
#                     task_id = str(k)
#                     print(f' ----- Evaluating using task_id: {task_id} ----- ')
#                     print(f' ----- labels (global): {labels} ----- ')
#                     if task_id in model.net.classifiers:
#                         model.net.classifiers[task_id].to(model.device)
#                     outputs = model.net(inputs, task_id=task_id)
                    
#                     # Adjust global labels to local if needed...
#                     if hasattr(model, 'dataset') and hasattr(model.dataset, 'N_CLASSES_PER_TASK'):
#                         offset = int(task_id) * model.dataset.N_CLASSES_PER_TASK
#                         print(f"Adjusting labels in evaluation with offset: {offset}")
#                         labels = labels - offset
#                 else:
#                     # Fallback case.
#                     if 'class-il' not in model.COMPATIBILITY:
#                         outputs = model(inputs, k)
#                     else:
#                         outputs = model(inputs)

#                 # Compute predictions and update accuracy.
#                 if isinstance(outputs, tuple):
#                     outputs = outputs[0]

#                 _, pred = torch.max(outputs.data, 1)
#                 correct += torch.sum(pred == labels).item()
#                 total += labels.shape[0]

#                 if dataset.SETTING == 'class-il':
#                     mask_classes(outputs, dataset, k)
#                     _, pred = torch.max(outputs.data, 1)
#                     correct_mask_classes += torch.sum(pred == labels).item()

#         acc = correct / total * 100
#         print(f'Task {k} Accuracy: {acc:.2f}%')
        
#         accs.append(acc if 'class-il' in model.COMPATIBILITY else 0)
#         accs_mask_classes.append(correct_mask_classes / total * 100)

#     model.net.train(status)
#     return accs, accs_mask_classes


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    Loads task-specific adapter weights before evaluation if using adapters.
    """
    print(f' ***** STARTING evaluate ****** ')
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
            
        # Load task-specific adapter weights if using adapters
        if hasattr(model.args, 'use_adapter') and model.args.use_adapter:
            adapter_dir = os.path.join(
                model.args.output_dir,
                "adapters",
                dataset.SETTING,
                dataset.NAME,
                model.NAME,
                model.args.experiment_id
            )
            adapter_path = os.path.join(adapter_dir, f'adapter_task{k}.pt')
            if os.path.exists(adapter_path) and hasattr(model.net, 'set_adapter_weights'):
                adapter_state = torch.load(adapter_path)
                model.net.set_adapter_weights(str(k), adapter_state)
                print(f"***** Loaded adapter weights for task {k} from {adapter_path} *****")
            
        # Loop over the test set for task k
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)

                # For multihead, pass the correct task_id for evaluation
                if hasattr(model.net, 'multihead') and model.net.multihead:
                    task_id = str(k)
                    print(f' ----- Evaluating using task_id: {task_id} ----- ')
                    print(f' ----- labels (global): {labels} ----- ')
                    if task_id in model.net.classifiers:
                        model.net.classifiers[task_id].to(model.device)
                    outputs = model.net(inputs, task_id=task_id)
                    
                    # Adjust global labels to local if needed
                    if hasattr(model, 'dataset') and hasattr(model.dataset, 'N_CLASSES_PER_TASK'):
                        offset = int(task_id) * model.dataset.N_CLASSES_PER_TASK
                        print(f"Adjusting labels in evaluation with offset: {offset}")
                        labels = labels - offset
                else:
                    # Fallback case
                    if 'class-il' not in model.COMPATIBILITY:
                        outputs = model(inputs, k)
                    else:
                        outputs = model(inputs)

                # Compute predictions and update accuracy
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        acc = correct / total * 100
        print(f'Task {k} Accuracy: {acc:.2f}%')
        
        accs.append(acc if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes



def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at handd
    :param args: the arguments of the current execution
    """
    print(args)

    # if not args.nowand:
    #     assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    #     print(args.wandb_project)
    #     print(args.wandb_entity)
    #     wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    #     args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME, args.output_dir, args.experiment_id)

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)
        model.writer = tb_logger.loggers[dataset.SETTING]

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)
            # Added for FWT
            model.random_results_class = random_results_class
            print(f' ----- random_results_class: {random_results_class} -----')
            model.random_results_task = random_results_task
            print(f' ----- random_results_task: {random_results_task} -----')

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        print(f' ----- 1. t: {t} ----- ')
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            print(f' ----- 1.1 begin_task ----- ')
            model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            print(f' ----- 1.2 evaluate ----- ')
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)
        print(f' ===== model.args.n_epochs : {model.args.n_epochs} ===== ')
        for epoch in range(model.args.n_epochs):
            print(f' ===== 2. epoch: {epoch} ===== ')
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                print(f' ===== 2.1 i: {i} ===== ')
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    print(f' ===== 2.2 logits ===== ')
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    # print(f' ----- labels: {labels} ----- ')
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    print(f' ===== 2.3 data ===== ')
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, str(t))
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if hasattr(model, 'end_epoch'):
                model.end_epoch(dataset)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            print(f' ----- args.nowand : {args.nowand} ----- ')
            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}
            wandb.log(d2)

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()

    if args.tensorboard:
        tb_logger.close()


# ========== Adapter Training ==========
def train_adapter(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)
    # Move entire model to device
    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME, args.output_dir, args.experiment_id)

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)
        model.writer = tb_logger.loggers[dataset.SETTING]

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    # Initialize random results if needed
    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        
        # Initialize task-specific heads for all tasks before evaluation
        if hasattr(model.net, 'multihead') and model.net.multihead:
            for t in range(dataset.N_TASKS):
                model.net.add_task_classifier(str(t), dataset.N_CLASSES_PER_TASK)
                # Explicitly move new classifier to device
                model.net.classifiers[str(t)].to(model.device)
                _, _ = dataset_copy.get_data_loaders()
        
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            # Ensure model is in eval mode before random evaluation
            model.net.eval()
            random_results_class, random_results_task = evaluate(model, dataset_copy)
            model.random_results_class = random_results_class
            model.random_results_task = random_results_task
            model.net.train()

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        print(f'Starting training on task {t}')
        
        # Add new classifier head for this task if using multihead
        if hasattr(model.net, 'multihead') and model.net.multihead:
            task_id = str(t)  # Convert task number to string ID
            print(f' ----- task_id: {task_id} ----- ')
            num_classes = dataset.N_CLASSES_PER_TASK  # Always use fixed number of classes
            print(f' ----- num_classes: {num_classes} ----- ')
            if not hasattr(model.net, f'classifier_{task_id}'):
                model.net.add_task_classifier(task_id, num_classes)
                print(f'Added new classifier head for task {task_id} with {num_classes} classes.')
                model.net.classifiers[task_id].to(model.device)

        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
            
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        scheduler = dataset.get_scheduler(model, args)

        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
                
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                    
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    print(f' ----- IF labels: {labels} ----- ')
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    # print(f' ----- ELSE labels: {labels} ----- ')
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    # Pass task_id to meta_observe for multihead models
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, str(t))

                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if hasattr(model, 'end_epoch'):
                model.end_epoch(dataset)

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        # Evaluate using the appropriate task head
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            print(f' ----- args.nowand : {args.nowand} ----- ')
            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}
            wandb.log(d2)

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()

    if args.tensorboard:
        tb_logger.close()

# ========== Class Adapter Training ==========
def train_class_adapter(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
    """
    Trains classifier adapters at the class level based on groups of classes.
    
    The groups are defined as:
      – Group "group0": classes [0, 8, 9]  → 3 classes
      – Group "group1": classes [1, 2]     → 2 classes 
      – Group "group2": classes [3, 4, 5, 6, 7] → 5 classes

    For each group, a new classifier adapter is initialized (if not already present)
    and trained on the subset of data corresponding to its classes. After training,
    the adapter weights for that group are saved.
    """
    import os  # in case not yet imported

    print(args)
    # Move entire model to device
    model.net.to(model.device)
    
    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME, args.output_dir, args.experiment_id)
    
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)
        model.writer = tb_logger.loggers[dataset.SETTING]
    
    progress_bar = ProgressBar(verbose=not args.non_verbose)
    
    # Define the class groups mapping.
    class_groups = {
        "group0": [0, 8, 9],
        "group1": [1, 2],
        "group2": [3, 4, 5, 6, 7]
    }
    
    # Optional random evaluation with a dataset copy.
    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        if hasattr(model.net, 'multihead') and model.net.multihead:
            for group_id, class_list in class_groups.items():
                if not hasattr(model.net, f'classifier_{group_id}'):
                    model.net.add_task_classifier(group_id, len(class_list))
                    model.net.classifiers[group_id].to(model.device)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            model.net.eval()
            random_results_class, random_results_task = evaluate(model, dataset_copy)
            model.random_results_class = random_results_class
            model.random_results_task = random_results_task
            model.net.train()

    # Helper function to evaluate the adapter for one class group.
    # def evaluate_class_group(group_id: str, class_list: list) -> float:
    #     # Get the test loader (assumes get_data_loaders returns (train_loader, test_loader))
    #     _, test_loader = dataset.get_data_loaders()
    #     model.net.eval()
    #     correct, total = 0.0, 0.0
    #     for data in test_loader:
    #         # Expecting data to be a tuple (inputs, labels, not_aug_inputs) [possibly with logits]
    #         inputs, labels, *rest = data
    #         inputs = inputs.to(model.device)
    #         labels = labels.to(model.device)
    #         # Create a mask for examples that belong to the current class group.
    #         mask = torch.tensor([l.item() in class_list for l in labels],
    #                             dtype=torch.bool, device=model.device)
    #         if mask.sum() == 0:
    #             continue
    #         filtered_inputs = inputs[mask]
    #         filtered_labels = labels[mask]
    #         # Remap filtered global labels into local indices for the group.
    #         new_labels = torch.tensor([class_list.index(l.item()) for l in filtered_labels],
    #                                   device=model.device)
    #         # Forward using the adapter for the current group.
    #         outputs = model.net(filtered_inputs, task_id=group_id)
    #         _, preds = torch.max(outputs.data, 1)
    #         correct += (preds == new_labels).sum().item()
    #         total += new_labels.size(0)
    #     acc = 100 * correct / total if total > 0 else 0.0
    #     print(f"Accuracy for class group {group_id}: {acc:.2f}%")
    #     model.net.train()
    #     return acc
    
    # Loop over each class group
    for group_id, class_list in class_groups.items():
        print(f"Starting training on class group {group_id} with classes {class_list}")
        # If using multihead mode, add a new classifier adapter head if not already present.
        if hasattr(model.net, 'multihead') and model.net.multihead:
            if not hasattr(model.net, f'classifier_{group_id}'):
                model.net.add_task_classifier(group_id, len(class_list))
                print(f"Added new classifier adapter for class group {group_id} with {len(class_list)} classes.")
                model.net.classifiers[group_id].to(model.device)
        
        # Get the data loaders.
        train_loader, _ = dataset.get_data_loaders()
        
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        
        scheduler = dataset.get_scheduler(model, args)
        
        # Train for the prescribed number of epochs.
        for epoch in range(model.args.n_epochs):
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                # Check whether the dataset returns logits along with inputs.
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    # Filter the batch to keep only examples whose label is in the current class group.
                    mask = torch.tensor([l.item() in class_list for l in labels], dtype=torch.bool)
                    if mask.sum().item() == 0:
                        continue
                    inputs = inputs[mask].to(model.device)
                    labels = labels[mask].to(model.device)
                    not_aug_inputs = not_aug_inputs[mask].to(model.device)
                    logits = logits[mask].to(model.device)
                    new_labels = torch.tensor([class_list.index(l.item()) for l in labels],
                                              dtype=torch.long).to(model.device)
                    loss = model.meta_observe(inputs, new_labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    mask = torch.tensor([l.item() in class_list for l in labels], dtype=torch.bool)
                    if mask.sum().item() == 0:
                        continue
                    inputs = inputs[mask].to(model.device)
                    labels = labels[mask].to(model.device)
                    not_aug_inputs = not_aug_inputs[mask].to(model.device)
                    new_labels = torch.tensor([class_list.index(l.item()) for l in labels],
                                              dtype=torch.long).to(model.device)
                    # Pass the group identifier as the task id.
                    loss = model.meta_observe(inputs, new_labels, not_aug_inputs, str(group_id))
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, group_id, loss)
            if hasattr(model, 'end_epoch'):
                model.end_epoch(dataset)
            if scheduler is not None:
                scheduler.step()
        
        if hasattr(model, 'end_task'):
            model.end_task(dataset)
        
        # Evaluate on the test set using only examples from this class group.
        group_acc = evaluate_class_group(group_id, class_list)
        if not args.disable_log:
            logger.log([group_acc])
        
        # Save the classifier adapter weights after training is done.
        adapter_dir = os.path.join(
            args.output_dir,
            "class_adapters",
            dataset.SETTING,
            dataset.NAME,
            model.NAME,
            args.experiment_id
        )
        os.makedirs(adapter_dir, exist_ok=True)
        if hasattr(model.net, 'get_adapter_weights'):
            adapter_state = model.net.get_adapter_weights(str(group_id))
            if adapter_state is not None:
                adapter_path = os.path.join(adapter_dir, f'class_adapter_{group_id}.pt')
                torch.save(adapter_state, adapter_path)
                print(f"Saved classifier adapter for class group {group_id} to {adapter_path}")
    
    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url() if wandb is not None else ""
            if wandb is not None:
                wandb.log(d)
    if not args.nowand and wandb is not None:
        wandb.finish()
    if args.tensorboard:
        tb_logger.close()
