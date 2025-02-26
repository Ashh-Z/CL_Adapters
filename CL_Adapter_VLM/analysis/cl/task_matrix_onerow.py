import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# lst_colors = [
#     '#f9dbbd',
#     '#ffa5ab',
#     '#da627d',
#     '#a53860',
#     '#450920',
# ]
lst_colors = [
    # '#f9dbbd',
    "#ffe5d9",
    "#ffcfd2",
    '#ffa5ab',
    '#da627d',
    '#a53860',
    # '#450920',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

dataset = 'domainnet'

if dataset == 'cifar10':
    lst_methods = {
        'ER': '/data/output-ai/fahad.sarfraz/SYNERgy/er_cifar10/results/class-il/seq-cifar10/er/er-cifar10-%s-param-v1-r3/',
        'DER++': '/data/output-ai/fahad.sarfraz/SYNERgy/derpp_cifar10/results/class-il/seq-cifar10/derpp/derpp-cifar10-%s-param-v1-r3/',
        'cls-er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/cil_er/results/class-il/seq-cifar10/dualmeanerv6/er-c10-%s-param-v1-s-0_stable_ema_model/',
        'aux': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-%s-seq-cifar10-kl1.0-0.1-up0.2-s2_ema_net1/',
    }
    num_tasks = 5
    annot = True
elif dataset == 'cifar100':
    lst_methods = {
        'ER': 'er/results/class-il/seq-tinyimg/er/er-tinyimg-b-%s-r-0/task_performance.txt',
        'DER++': 'der/results/class-il/seq-tinyimg/derpp/der-tinyimg-b-%s-r-0/task_performance.txt',
    }
    num_tasks = 5
elif dataset == 'domainnet':
    lst_methods = {
        'DER++': '/data/output-ai/shruthi.gowda/continual/domainNet/base/base_tiny_params/tiny_param/results/domain-il/domain-net/derpp/cll-derpp-%s-domain-netv2-s0',
        'ER': '/data/output-ai/shruthi.gowda/continual/domainNet/base/base_tiny_params/tiny_param/results/domain-il/domain-net/er/cll-er-%s-domain-netv2-s0',
        'cls-er': '/data/output-ai/shruthi.gowda/continual/base/clser/results/domain-il/domain-net/clser/clser-%s-domain-net-lr0.05-0.080.05-mod-s0_stable_model',
        'aux': '/data/output-ai/shruthi.gowda/continual/domainNet/method/results/domain-il/domain-net/derpp_mm_eman1/cll-%s-domain-netv2-lr0.03-l20.10.01-em0.06-s2_ema_net1',

    }
    num_tasks = 6

n_rows, n_cols = 1, 4 #2, 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 9), sharey=True, sharex=True)

annot = True
fmt = '.1f'
# fmt = '%d'
font = 18

lst_method = ['ER', 'DER++', 'cls-er', 'aux']
buffer_size = 500

# Get Max and Min
v_max = 0
v_min = 1000
for n, method in enumerate(lst_method):
    perf_path = os.path.join(lst_methods[method] % buffer_size, 'task_performance.txt')

    np_perf = np.loadtxt(perf_path)
    max, min = np_perf.max(), np_perf.min()

    if v_max < max:
        v_max = max
    if v_min > min:
        v_min = min


x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

x_labels = ['real', 'clip', 'igraph', 'paint', 'sketch', 'qdraw']
y_labels = [f"After {i}" for i in x_labels]

k = 0
for n, method in enumerate(lst_method):
    perf_path = os.path.join(lst_methods[method] % buffer_size, 'task_performance.txt')

    np_perf = np.loadtxt(perf_path)
    matrix = np.triu(np.ones_like(np_perf)) - np.identity(np_perf.shape[0])

    if n == 3:
        #cbar_ax = fig.add_axes([0.91, 0.18, 0.02, 0.6])
        im = sns.heatmap(np_perf, ax=ax[n], vmax=v_max, vmin=v_min, mask=matrix, annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": 16}) #, cbar_ax=cbar_ax)
    else:
        im = sns.heatmap(np_perf, ax=ax[n], vmax=v_max, vmin=v_min, mask=matrix,  annot=annot, cmap=custom1, alpha=0.85, cbar=False, fmt=fmt, linewidths=.5, annot_kws={"size": 16})
    ax[n].set_xticks(np.arange(len(x_labels)) + 0.5)
    ax[n].set_yticks(np.arange(len(y_labels)) + 0.5)
    ax[n].set_xticklabels(x_labels, ha='center', fontsize=21, rotation=25)
    ax[n].set_yticklabels(y_labels, rotation=0, va='center', fontsize=21)
    ax[n].set_aspect('equal', adjustable='box')

    ax[n].axhline(y=0, color='k', linewidth=1)
    ax[n].axhline(y=np_perf.shape[1], color='k', linewidth=2)
    ax[n].axvline(x=0, color='k', linewidth=1)
    ax[n].axvline(x=np_perf.shape[1], color='k', linewidth=2)


font = 24
ax[0].set_title('ER', fontsize=font)
ax[1].set_title('DER++', fontsize=font)
ax[2].set_title('CLS-ER', fontsize=font)
ax[3].set_title('DUCA', fontsize=font)
# ax[1][2].axis('off')
# ax[1][0].set_position([0.24,0.125,0.228,0.343])
# ax[1][1].set_position([0.55,0.125,0.228,0.343])


# fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.show()

fig.savefig(f'/volumes2/continual_learning/paper/icml/task_dom_500.png', bbox_inches='tight')
fig.savefig(f'/volumes2/continual_learning/paper/icml/task_dom_500_n.pdf', bbox_inches='tight')
