import os
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from glob import glob
from norm_datasets.dataset import DATASETS
from backbone.ResNet_mam_llm import resnet18mamllm
from backbone.ResNet_mam import resnet18mam
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# =============================================================================
# Robustness Evaluation
# =============================================================================
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  device,
                  save_imgs = False,
                  ind = 0
                  ):

    out = model(X)
    out = out[0] if isinstance(out, tuple) else out
    out = out['logits'] if isinstance(out, dict) else out
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            out = model(X_pgd)
            out = out[0] if isinstance(out, tuple) else out
            out = out['logits'] if isinstance(out, dict) else out

            loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    out = model(X_pgd)
    out = out[0] if isinstance(out, tuple) else out
    out = out['logits'] if isinstance(out, dict) else out

    err_pgd = (out.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_robustness(
    model,
    data_loader,
    epsilon,
    num_steps,
    step_size,
    random=True,
    device='cuda',
    save_imgs = False
):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    ind = 0

    for data, target in tqdm(data_loader, desc='robustness'):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, epsilon, num_steps, step_size, random, device, save_imgs, ind)
        robust_err_total += err_robust
        natural_err_total += err_natural
        ind += 65
        # if save_imgs:
        #     for i in range(len(adv_imgs)):
        #         from data.utils import plot
        #         orig = X[i].squeeze().detach().cpu().numpy()
        #         plot(orig, 'orig_%s' % i)
        #         adv = adv_imgs[i].squeeze().detach().cpu().numpy()
        #         plot(adv, 'adv_%s' % i)

    nat_err = natural_err_total.item()
    successful_attacks = robust_err_total.item()
    total_samples = len(data_loader.dataset)

    rob_acc = (total_samples - successful_attacks) / total_samples
    nat_acc = (total_samples - nat_err) / total_samples

    print('=' * 30)
    print(f"Adversarial Robustness = {rob_acc * 100} % ({total_samples - successful_attacks}/{total_samples})")
    print(f"Natural Accuracy = {nat_acc * 100} % ({total_samples - nat_err}/{total_samples})")

    return nat_acc, rob_acc


class arg_class():
    def __init__(self, dataset, model_architecture, num_classes):
        self.dataset = dataset
        self.model_architecture = model_architecture
        self.num_classes = num_classes

lst_models = glob(r'/volumes1/vlm-cl/normal_cls/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/dytox_cls/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/results_lllm/snel2/*/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/results_lllm/results_lllm/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/llm_wt/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/llm_wt/old/*/model.ph')
lst_exp = [
    # 'normal-resnet18mam-cifar10-desc--e-100-s-42',
    # 'revproj-vlm-resnet18mam-cifar10-desc-e-100-l-sim-20.0-text-sent_transf-s-42',
    # 'normal-resnet18mamllm-cifar10-desc--l0.001-e-150-s-42',

    # 'ex-resnet18mam-cifar10-tsent_transf_large-lr0.1-w0.0005-ep100-l-15.0-s-1',
    'ex-resnet18mam-llmrandom-cifar10-lr0.1-w0.0-e100-s-0',
    'ex-resnet18mam-llmcode_lm-cifar10-lr0.1-w0.0-e100-s-0',
    # 'ix-cifar10-resnet18mamllm-sent_transf_large-lr0.0001-w0.1-ep100-s-1',
    # 'ix-cifar10-resnet18mamllm-code_lm-lr0.0001-w0.1-ep100-s-1',
    # 'ix-resnet18mamllm-llmrandperm-random-cifar10-lr0.0001-w0.1-e100-s0-test'

    # 'ex-resnet18mam-llmrandom-cifar10-lr0.1-w0.0005-e100-[10-s0'
]

results = {
    'id': [],
    'model' : [],
    'eps': [],
    'num_steps': [],
    'accuracy': [],
    'robustness': []
}

eps_lst = [0.25/255, 0.5/255, 1/255, 2/255, 4/255, 8/255]
lst_pgd_steps = [10] #, 20]
count = 0
for model_path in lst_models:
    path_tokens = model_path.split('/')
    # dataset = path_tokens[-4]
    exp_id = path_tokens[-2]
    if exp_id in lst_exp:
        print('*' * 30)
        print(exp_id)
        print('*' * 30)
        # Get the data
        data = ('cifar10', 2, '/volumes1/datasets/cifar/CIFAR10')
        dataset_args = {"data_path": data[2], "arch": 'resnet18mam'}
        dataset = DATASETS[data[0]](**dataset_args)

        transform_test = transforms.Compose([transforms.ToTensor()])
        testset = dataset.get_dataset('test', None, transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=64, shuffle=False, num_workers=4
        )
        if 'ex' in exp_id:
            model = resnet18mam(dataset.NUM_CLASSES).to(device)
        elif 'code_lm' in exp_id:
            llm_block = 'code_lm'
            model = resnet18mamllm(dataset.NUM_CLASSES, llm_block=llm_block).to(device)
        elif 'large' in exp_id or 'distil' in exp_id:
            llm_block = 'sent_transf_large'
            model = resnet18mamllm(dataset.NUM_CLASSES, llm_block=llm_block).to(device)
        elif 'random' in exp_id:
            llm_block = 'sent_transf'
            model = resnet18mamllm(dataset.NUM_CLASSES, llm_block=llm_block, llm_pretrain='False').to(device)
        else:
            model = resnet18mam(dataset.NUM_CLASSES).to(device)

        state_dict = torch.load(model_path)['state_dict']
        try:
            model.load_state_dict(state_dict) #, strict=False)
        except RuntimeError as err:
            print(err)
            continue
        model = model.cuda()

        for eps in eps_lst:
            for num_steps in lst_pgd_steps:
                test_acc, test_rob = eval_adv_robustness(
                    model,
                    test_loader,
                    eps, #0.007
                    num_steps,
                    0.003,
                    random=True,
                    device='cuda' if cuda else 'cpu',
                    save_imgs = False
                )
                print('Accuracy:', test_acc)
                print(f'PGD-{num_steps}', test_rob)
                results['id'].append(exp_id)
                results['model'].append(os.path.basename(model_path))
                results['eps'].append(eps)
                results['num_steps'].append(num_steps)
                results['accuracy'].append(test_acc)
                results['robustness'].append(test_rob)

        count+=1
        df = pd.DataFrame(results)
        df.to_csv('/volumes1/vlm-cl/paper/adv_eval_llm_1.csv')

print(count)