import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
# import sys
# sys.path.insert(0, './backbone')

device = 'cuda'

dataset = 'cifar10'
lst_methods = {
    'der': '/data/output-ai/fahad.sarfraz/SYNERgy/derpp_cifar10/task_models/seq-cifar10/derpp-cifar10-200-param-v1-r0/task_%s_model.ph',
    # 'er': '/data/output-ai/fahad.sarfraz/SYNERgy/er_cifar10/task_models/seq-cifar10/er-cifar10-200-param-v1-r0/task_%s_model.ph',
    # 'cls-er': '/data/output-ai/fahad.sarfraz/SYNERgy/analysis_models/cifar10/cil_er/task_models/seq-cifar10/er-c10-200-param-v1-s-6/task_%s_stable_model.ph',
    # 'aux': '/data/output-ai/shruthi.gowda/continual/clmm_derpp_v2n1/results/class-il/seq-cifar10/derpp_mm_eman1_model/derppema-n1m-200-seq-cifar10-kl1.0-0.1-up0.2-s2_ema_net1/ema_net1_%s.pth'
}
#
# dataset = 'cifar100'
# lst_methods = {
#     'der': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/derpp/cll-derpp-200-seq-cifar100-s0/net_%s.pth',
#     'er': '/data/output-ai/shruthi.gowda/continual/baseline/results/class-il/seq-cifar100/er/cll-er-200-seq-cifar100-s0/net_%s.pth',
#     'cls-er':'/data/output-ai/fahad.sarfraz/baseline/task_models/seq-cifar100/c100-5-200-param-v1-s-0/task_%s_stable_model.ph',
#     # 'cls-er': '/data/output-ai/fahad.sarfraz/lll_baselines/task_models/seq-cifar100/c100-5-200-param-v4-0.05-0.1s-0/stable_model.ph',
#     'aux': '/data/output-ai/shruthi.gowda/continual/cifar100/results/class-il/seq-cifar100/derpp_mm_eman1/cll-cif100200-a0.15b0.5-lr0.03-l20.10.01-up0.06-urgent-s0_ema_net1/ema_net1_%s.pth'
# }

# dataset = 'domainnet'
# lst_methods = {
#     'der': '/data/output-ai/shruthi.gowda/continual/domainNet/base/base_tiny_params/tiny_param/results/class-il/domain-net/er/cll-er-500-domain-netv2-mod-s0/net_%s.pth',
#     'er': '/data/output-ai/shruthi.gowda/continual/domainNet/base/base_tiny_params/tiny_param/results/class-il/domain-net/derpp/cll-derpp-500-domain-netv2-s0/net_%s.pth',
#     'cls-er':'/data/output-ai/shruthi.gowda/continual/base/clser/results/class-il/domain-net/clser/clser-500-domain-net-lr0.05-0.080.05-mod-s0/net_%s.pth',
#     'aux': '/data/output-ai/shruthi.gowda/continual/domainNet/method/results/class-il/domain-net/derpp_mm_eman1/cll-500-domain-netv2-lr0.03-l20.10.01-em0.06-urgent-s0_ema_net1/ema_net1_%s.pth'
# }

method = 'aux'
buffer_size = 500
num_tasks = 5
task_id = 4

# lst_models = ['er', 'der', 'cls-er', 'aux']
lst_models = ['der']


def get_normalized_params(model):
    lst_params = []
    for param in model.parameters():
        lst_params.append(param.view(-1) / param.max())

    return torch.cat(lst_params)

lst_diff = []
for method in lst_models:

    lst_params = []
    # model_path = lst_methods[method]
    if method == 'cls-er':
        model_path = lst_methods[method] % (task_id + 1)
    else:
        model_path = lst_methods[method] % (task_id)

    model = torch.load(model_path).to(device)
    #lst_params.append(model.get_params() / model.get_params().max())
    lst_params.append(list(model.parameters()))

    from torchsummary import summary
    summary(model, (3,64,64))