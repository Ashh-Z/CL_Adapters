from __future__ import print_function
import os
import torch
import torch.nn.functional as F
from matplotlib.offsetbox import AnchoredText
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from norm_datasets.dataset import DATASETS
from backbone.ResNet_mam_llm import resnet18mamllm
from backbone.ResNet_mam import resnet18mam

def eval_calibration(model, device, data_loader, axes):
    n_bins = 15
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    lst_logits = []
    lst_labels = []

    with torch.no_grad():
        for input, label in data_loader:
            input, label = input.to(device), label.to(device)
            logits = model(input)
            lst_logits.append(logits.detach().cpu())
            lst_labels.append(label.detach().cpu())

    logits = torch.cat(lst_logits)
    labels = torch.cat(lst_labels)
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    lst_acc_in_bin = []
    lst_conf_in_bin = []
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            lst_acc_in_bin.append(accuracy_in_bin.item())
            lst_conf_in_bin.append(avg_confidence_in_bin.item())
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            lst_acc_in_bin.append(0)
            lst_conf_in_bin.append(0)
    ece *= 100
    print('ECE: %s' % ece.item())
    col = (237, 129, 121)
    col = np.array(col) / 255
    col = tuple(col.tolist())
    x_axis = np.array(list(range(0, n_bins))) / n_bins
    axes.bar(x_axis, x_axis, align='edge', width=0.075, facecolor=(1, 0, 0, 0.3), edgecolor=col, hatch='//', label='Gap')
    axes.bar(x_axis, lst_acc_in_bin, align='edge', width=0.075, facecolor=(0, 0, 1, 0.5), edgecolor='blue', label='Outputs')
    x_axis = np.array(list(range(0, n_bins + 1))) / n_bins
    axes.plot(x_axis, x_axis, '--', color='k')
    axes.axis('equal')
    # axes.xlim(0, 1)
    # axes.ylim(0, 1)
    # axes.gca().set_aspect('equal', adjustable='box')
    # axes.set_ylabel('Accuracy', labelpad=11, fontsize=17, color='k')
    axes.set_xlabel('Confidence', fontsize=19,)
    anchored_text = AnchoredText('ECE=%.2f' % ece, loc='upper left', prop=dict(fontsize=16))
    axes.add_artist(anchored_text)
    # plt.yticks(fontsize=12)
    # plt.xticks(fontsize=12)
    # axes.set_xticklabels(axes.get_xticklabels(), fontsize=14)
    # axes.set_xticklabels(axes.get_xticklabels(), fontsize=14)
    return ece.item()


# =============================================================================
# Load Dataset
# =============================================================================
lst_models = glob(r'/volumes1/vlm-cl/snel/results_cls/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/*/model.ph')
lst_exp = [
    'tinyimagenet-lr0.03-ep100-s-0',
    # 'vlm-resnet18mam-tinyimagenet-desc-e-100-l-0.05-200.0-text-sent_transf-s-0',
    'ex-vlm-tinyimagenet-tsent_transf-lr0.05-w0.001-ep100-l-150-s-0',
    # 'ix-tinyimagenet-resnet18mamllm-lr0.03-w0.01-ep200-s-0',
    'ix-tinyimagenet-resnet18mamllm-lr0.01-w0.01-ep100-s-0'
]

TRANSFORM = transforms.Compose(
        [transforms.ToTensor()])
         # transforms.Normalize((0.4802, 0.4480, 0.3975),
         #                      (0.2770, 0.2691, 0.2821))])

dataset = 'tinyimagenet'
dataset_path = '/volumes1/datasets/tiny-imagenet-200'
class arg_class():
    def __init__(self, dataset, ft_prior):
        self.dataset = dataset
        self.ft_prior = ft_prior
# =============================================================================
# Evaluate Calibration Plots
# =============================================================================
device = 'cuda'
NUM_CLASSES = 200
batch_size = 64
n_rows, n_cols = 1, 3
ax = [0,1,2]
name = ['Base', 'ExLG', 'IxLG']
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5), sharey=True)
i = 0
for model_path in lst_models:
    path_tokens = model_path.split('/')
    exp_id = path_tokens[-2]
    if exp_id in lst_exp:
        print('*' * 30)
        print(exp_id)
        print('*' * 30)

        if 'llm' in exp_id:
            model = resnet18mamllm(NUM_CLASSES).to(device)
        else:
            model = resnet18mam(NUM_CLASSES).to(device)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict)

        model = model.cuda()
        data = (dataset, 200, dataset_path)
        testset = DATASETS[data[0]](data[2])
        test_dataset = testset.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        eval_calibration(model, 'cuda', test_loader, axes[ax[i]])
        axes[ax[i]].set_title(name[i], fontdict={'fontsize': 20})
        axes[2].legend(fontsize=16)

        i+=1

axes[0].set_ylabel('Accuracy', labelpad=11, fontsize=19, color='k')
plt.show()
fig.savefig('/volumes1/vlm-cl/paper/calib.png', bbox_inches='tight')
