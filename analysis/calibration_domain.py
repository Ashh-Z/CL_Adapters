from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from scipy.stats import norm

batch_size = 64
font = 12
# matplotlib.rc('font', **font)
index = 7
TRANSFORM = transforms.Compose(
    [
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2615))
     ]
)

# CIFAR 10
TRANSFORM_AUX = transforms.Compose(
    [
         transforms.ToTensor(),
     ]
)

from datasets.domain_net import ImageFilelist

base = '/data/input-ai/datasets/'
data_path = base + 'domain_net'
DOMAIN_LST = ['real', 'clipart', 'infograph', 'painting', 'sketch', 'quickdraw']
annot_path = os.path.join(base + 'domain_net_cl', 'version2')
data_loaders = []
for i in range(6):
    dataset = ImageFilelist(
                root=data_path,
                flist=os.path.join(annot_path, DOMAIN_LST[i] + "_test.txt"),
                transform=transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor()]),
                )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    data_loaders.append(data_loader)
data_loader_auxs = data_loaders
lst_methods = {
    'er': '/data/output-ai/shruthi.gowda/continual/domain_net/base/tiny_param/results/class-il/domain-net/er/cll-er-500-domain-netv2-mod-s0/net_final.pth',
    'der': '/data/output-ai/shruthi.gowda/continual/domain_net/base/tiny_param/results/class-il/domain-net/derpp/cll-derpp-500-domain-netv2-s0/net_final.pth',
    'cls-er':'/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/domain-net/clser/clser-500-domain-net-lr0.05-0.080.05-mod-s0/net_final.pth',
    'aux': '/data/output-ai/shruthi.gowda/continual/domain_net/results/class-il/domain-net/derpp_mm_eman1/cll-500-domain-netv2-lr0.03-l20.10.01-em0.06-urgent-s0_ema_net1/ema_net1_final.pth'
}

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
    anchored_text = AnchoredText('ECE=%s' % ece, loc='lower right', prop=dict(fontsize=8))
    ax.add_artist(anchored_text)

    # plt.tight_layout()
    # return fig

def eval_calib(test_loaders, names, axes, ind):
    labels = []
    logits = []

    for k, test_loader in enumerate(test_loaders):
        logits1 = []
        labels1 = []
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

# Configuration
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

n_bins = 10
ece = ECE(n_bins)

fig, axes = plt.subplots(nrows=1, ncols=4, squeeze=True, figsize=(11, 3))

dst = '/volumes2/continual_learning/paper/analysis/new'
lst_method = ['ER', 'DER++', 'cls-er', 'aux']
names = ['ER', 'DER++', 'cls-er', 'CCL']

ind = 0

model_path = lst_methods['er']
model = torch.load(model_path).to(device)
model.eval()
eval_calib(data_loaders, names, axes, 0)

model_path = lst_methods['der']
model = torch.load(model_path).to(device)
model.eval()
eval_calib(data_loaders, names, axes, 1)

model_path = lst_methods['cls-er']
model = torch.load(model_path).to(device)
model.eval()
eval_calib(data_loaders, names, axes, 2)

model_path = lst_methods['aux']
model = torch.load(model_path).to(device)
model.eval()
eval_calib(data_loader_auxs, names, axes, 3)


# axes[0].set_ylabel('Accuracy', fontsize=font)
# axes[1].set_ylabel('Accuracy', fontsize=font)
# axes[1].set_xlabel('Confidence', fontsize=font)
# labels and legend of second plot
# axes[0].legend(['Perfect Calibration', 'Output', 'Gap'], fontsize=15, frameon=False)
fig.savefig(os.path.join(dst, 'calib_dom_500.pdf'), dpi=300, bbox_inches='tight')
# plt.show()
