import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
# libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

lst_methods = {
    'Replay-CL': r'/volumes1/vlm-cl/baseline_final/results/class-il/seq-cifar10/er/er-resnet50mam-seq-cifar10-buf-200-s-42/task_performance.txt',
    'LG-Ex': r'/volumes1/vlm-cl/results_final/results/class-il/seq-cifar10/vl_er/revproj-vl_er-resnet18mam-seq-cifar10-desc-200--lr-0.03-l-sim-12.0-text-sent_transf-s-42/task_performance.txt',
    'LG-Ix': r'/volumes1/vlm-cl/fahad/all_results_29_09/results_implicit_cuda1/class-il/seq-cifar10/er/lgix-er-resnet18mamllm-sent_transf-seq-cifar10-10-b-200--l0.005-wd0.01-e-100-l0.005--s-42/task_performance.txt',
}

num_tasks = 10
annot = True

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

fmt = '.1f'
# fmt = '%d'
fontsize = 10

lst_method = ['Replay-CL', 'LG-Ex', 'LG-Ix']

n_rows, n_cols = 1, 3
fig, ax = plt.subplots(n_rows, n_cols, figsize=(13, 5), sharey=True, sharex=True)
# Get Max and Min
v_max = 0
v_min = 1000
for n, method in enumerate(lst_method):
    perf_path = lst_methods[method]

    np_perf = np.loadtxt(perf_path)
    max, min = np_perf.max(), np_perf.min()

    if v_max < max:
        v_max = max
    if v_min > min:
        v_min = min

x_labels = [f"T{i}" for i in range(1, num_tasks + 1)]
y_labels = [f"After T{i}" for i in range(1, num_tasks + 1)]

results = {}

for n, method in enumerate(lst_method):
    perf_path = lst_methods[method]

    np_perf = np.loadtxt(perf_path)

    plasticity = np.mean(np.diag(np_perf))
    stability = np.mean(np_perf[-1][:-1])
    tradeoff = (2 * stability * plasticity) / (stability + plasticity)

    results[method] = [plasticity, stability, tradeoff]

plt.figure(figsize=(5, 4))
N = len(results)
ind = np.arange(N)

# set width of bars
barWidth = 0.20

plt.bar(ind, results['Replay-CL'], barWidth, label='ER', color='#bfdbf7')
plt.bar(ind + barWidth, results['LG-Ex'], barWidth, label='ExLG', color='#1f7a8c')
plt.bar(ind + 2 * barWidth, results['LG-Ix'], barWidth, label='ImLG', color='#ffb703')

plt.xticks(ind + barWidth, ('Plasticity', 'Stability', 'Tradeoff'))
plt.legend()
plt.ylim([25, 100])

plt.savefig('/volumes1/vlm-cl/paper/ps.png', bbox_inches='tight', dpi=350)
plt.show()





