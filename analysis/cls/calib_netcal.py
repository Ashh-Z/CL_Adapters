from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
import matplotlib
from argparse import ArgumentParser
from norm_datasets.dataset import DATASETS
from backbone.ResNet_mam_llm import resnet18mamllm
from backbone.ResNet_mam import resnet18mam

from scipy.stats import norm

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


batch_size = 64
font = 20
# matplotlib.rc('font', **font)
index = 7

def plot_confidence_histogram(X, matched, histograms,bin_bounds, title_suffix, ece, axes):
    """ Plot confidence histogram and reliability diagram to visualize miscalibration for condidences only. """

    # get number of bins (self.bins has not been processed yet)
    n_bins = len(bin_bounds[0][0]) - 1
    median_confidence = [(bounds[0][1:] + bounds[0][:-1]) * 0.5 for bounds in bin_bounds]
    mean_acc, mean_conf = [], []
    for batch_X, batch_matched, batch_hist, batch_median in zip(X, matched, histograms, median_confidence):
        acc_hist, conf_hist, _, num_samples_hist = batch_hist
        empty_bins, = np.nonzero(num_samples_hist == 0)

        # calculate overall mean accuracy and confidence
        mean_acc.append(np.mean(batch_matched))
        mean_conf.append(np.mean(batch_X))

        # set empty bins to median bin value
        acc_hist[empty_bins] = batch_median[empty_bins]
        conf_hist[empty_bins] = batch_median[empty_bins]

        # convert num_samples to relative afterwards (inplace denoted by [:])
        num_samples_hist[:] = num_samples_hist / np.sum(num_samples_hist)

    # get mean histograms and values over all batches
    acc = np.mean([hist[0] for hist in histograms], axis=0)
    conf = np.mean([hist[1] for hist in histograms], axis=0)
    uncertainty = np.sqrt(np.mean([hist[2] for hist in histograms], axis=0))
    num_samples = np.mean([hist[3] for hist in histograms], axis=0)
    mean_acc = np.mean(mean_acc)
    mean_conf = np.mean(mean_conf)
    median_confidence = np.mean(median_confidence, axis=0)
    bar_width = np.mean([np.diff(bounds[0]) for bounds in bin_bounds], axis=0)

    # compute credible interval of uncertainty
    p = 0.05
    z_score = norm.ppf(1. - (p / 2))
    uncertainty = z_score * uncertainty

    # if no uncertainty is given, set variable uncertainty to None in order to prevent drawing error bars
    if np.count_nonzero(uncertainty) == 0:
        uncertainty = None

    # calculate deviation
    deviation = conf - acc

    # -----------------------------------------
    # plot data distribution histogram first
    # fig, axes = plt.subplots(1, squeeze=True, figsize=(7, 6))
    ax = axes
    # set title suffix if given
    # if title_suffix is not None:
    #     ax.set_title(title_suffix, fontsize=18)
    # else:
    #     ax.set_title('Reliability Diagram')

    # create two overlaying bar charts with bin accuracy and the gap of each bin to the perfect calibration
    ax.bar(median_confidence, height=acc, width=bar_width, align='center',
           edgecolor='black', color='lightseagreen', yerr=uncertainty, capsize=4)
    ax.bar(median_confidence, height=deviation, bottom=acc, width=bar_width, align='center',
           edgecolor='black', color='darkorange', alpha=0.6)

    # draw diagonal as perfect calibration line
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    ax.tick_params(axis='both', labelsize=13)
    # ax.set_ylabel('Accuracy', fontsize=font)

    from matplotlib.offsetbox import AnchoredText
    anchored_text = AnchoredText('ECE=%s' % ece, loc='lower right', prop=dict(fontsize=18))
    ax.add_artist(anchored_text)

    # plt.tight_layout()
    # return fig
# Configuration
use_cuda = True
set_random_seed(10)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

n_bins = 10
ece = ECE(n_bins)
# Load data
dataset = 'tinyimagenet'
dataset_path = '/volumes1/datasets/tiny-imagenet-200'
NUM_CLASSES = 200
# folders = ['/volumes2/colla/seq-cifar10/er/buf_500/20220130_193747_708396','/volumes2/colla/seq-cifar10/derpp/simclr/buf_500/20220130_193159_413535']
#fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(11, 4))
fig, axes = plt.subplots(nrows=2, ncols=1, squeeze=True, figsize=(5, 8))
# =============================================================================
# Load Dataset
# =============================================================================

lst_models = glob(r'/volumes1/vlm-cl/snel/old/results_cls/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/cifar/*/model.ph')
lst_exp = [
    'tinyimagenet-lr0.03-ep100-s-0',
    # 'vlm-resnet18mam-tinyimagenet-desc-e-100-l-0.05-200.0-text-sent_transf-s-0',
    'ex-vlm-tinyimagenet-tsent_transf-lr0.05-w0.001-ep100-l-150-s-0',
    # 'ix-tinyimagenet-resnet18mamllm-lr0.03-w0.01-ep200-s-0',
    'ix-tinyimagenet-resnet18mamllm-lr0.01-w0.01-ep100-s-0'
]
dst = '/volumes1/vlm-cl/paper'
names = ['Base', 'ExLG', 'IxLG']
ind = 0
for model_path in lst_models:
    path_tokens = model_path.split('/')
    # dataset = path_tokens[-4]
    exp_id = path_tokens[-2]
    if exp_id in lst_exp:

        data = (dataset, 200, dataset_path)
        testset = DATASETS[data[0]](data[2])
        test_dataset = testset.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        if 'llm' in exp_id:
            model = resnet18mamllm(NUM_CLASSES).to(device)
        else:
            model = resnet18mam(NUM_CLASSES).to(device)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()
        labels = []
        logits = []

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data[0].to(device)
                y = data[1].to(device)
                labels.append(y)
                confidence = model(x)
                confidence = confidence[0] if isinstance(confidence, tuple) else confidence
                logits.append(F.softmax(confidence, dim=1))

        labels = torch.cat(labels).cpu().numpy()
        logits = torch.cat(logits).cpu().numpy()
        ece_score = ece.measure(logits, labels)
        text = "%.2f" % (ece_score * 100)
        diagram = ReliabilityDiagram(n_bins)
        # acc, deviation, median_confidence, uncertainty = diagram.plot(logits, labels, text=text)
        # plot_figures(0, ind, axes[0][ind], acc, deviation, median_confidence, uncertainty, title)
        # rel_fig = diagram.plot(logits, labels, ece=text)
        # rel_fig.savefig(os.path.join(dst,'calib_{}.png'.format(ind)), bbox_inches='tight')
        X, matched, histograms, bin_bounds, title_suffix = diagram.plot(logits, labels)
        plot_confidence_histogram(X, matched, histograms, bin_bounds, names[ind], text, axes[ind])

        # ind+=1

# axes[0].set_ylabel('Accuracy', fontsize=font)
# axes[1].set_ylabel('Accuracy', fontsize=font)
axes[1].set_xlabel('Confidence', fontsize=font)
# labels and legend of second plot
# axes[0].legend(['Perfect Calibration', 'Output', 'Gap'], fontsize=15, frameon=False)
fig.savefig(os.path.join(dst, 'calib_imgnetr.png'), dpi=300, bbox_inches='tight')
# plt.show()
