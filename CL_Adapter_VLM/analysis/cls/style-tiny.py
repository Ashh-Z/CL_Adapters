import os
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from norm_datasets.dataset import DATASETS
from backbone.ResNet_mam_llm import resnet18mamllm
from backbone.ResNet_mam import resnet18mam
from torchvision import transforms
from torch.autograd import Variable

def validate(model, dataset, cuda=True, verbose=True):
    # set model mode as test mode.
    model_mode = model.training
    model.train(mode=False)

    # prepare the data loader and the statistics.
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4
    )
    total_tested = 0
    total_correct = 0

    for data, labels in data_loader:
        # test the model.
        data = Variable(data).cuda() if cuda else Variable(data)
        labels = Variable(labels).cuda() if cuda else Variable(labels)
        scores = model(data)
        scores = scores[0] if isinstance(scores, tuple) else scores
        _, predicted = torch.max(scores, 1)
        # update statistics.
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)

    # recover the model mode.
    model.train(mode=model_mode)

    # return the precision.
    precision = total_correct / total_tested
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


# =============================================================================
# Load Datasets
# =============================================================================
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

dataset_name = 'tinystyle'
dataset_path = '/volumes1/datasets/tiny-imagenet-200-stylized'

# List of models
# lst_models = glob(r'/volumes1/vlm-cl/snel/results_cls/*/model.ph')
# lst_models += glob(r'/volumes1/vlm-cl/normal_cls/*/model.ph')
lst_models = glob(r'/volumes1/vlm-cl/normal_cls/results_lllm/snel2/*/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/llm_wt/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/results_lllm/snel2/results_lllm_last/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/normal_cls/llm_wt/*/model.ph')

lst_exp = [
    # 'tinyimagenet-lr0.03-ep100-s-0',
    # 'tinyimagenet-lr0.03-ep200-s-0',
    # 'ex-vlm-tinyimagenet-tsent_transf-lr0.05-w0.001-ep100-l-150-s-0',
    # 'ex-vlm-tinyimagenet-tsent_transf-lr0.05-w0.001-ep100-l-150-s-1',
    # 'ex-vlm-tinyimagenet-tsent_transf-lr0.05-w0.001-ep100-l-150-s-2',
    # 'ix-tinyimagenet-resnet18mamllm-lr0.003-w0.1-ep100-s-4',
    # 'ix-tinyimagenet-resnet18mamllm-lr0.003-w0.1-ep100-s-6',
    # 'ix-tinyimagenet-resnet18mamllm-lr0.003-w0.1-ep100-s-3',

    'ex-resnet18mam-tinyimagenet-tsent_transf_large-lr0.03-w0.0-ep100-l-80.0-s-5',
    'ex-resnet18mam-llmrandom-tinyimagenet-lr0.03-w0.0005-e100-[80-s0',
    'ex-resnet18mam-tinyimagenet-tcode_lm-lr0.03-w0.01-ep100-l-80.0-s-5',
    'ix-tinyimagenet-resnet18mamllm-largelmdistil-lr0.003-w0.1-ep100-s-5',
    'ix-tinyimagenet-resnet18mamllm-code_lm-lr0.0001-w0.0-ep100-s-5',
    'ix-tinyimagenet-resnet18mamllm-code_lm-lr0.0001-w0.1-ep100-s-5',
    'ix-resnet18mamllm-llmrandperm-random-tinyimagenet-lr0.003-w0.1-e100-s0'

]

results = {
    'id': [],
    'accuracy': [],
    'alpha': []
}
batch_size = 64
NUM_CLASSES = 200
alpha_values = [0.2, 0.5, 0.7] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for alpha in alpha_values:
    print(f'Processing for alpha={alpha}')

    # Load the stylized TinyImageNet dataset for the current alpha
    alpha_path = "alpha_{}".format(alpha)
    testset = DATASETS[dataset_name](dataset_path, alpha_path)
    test_dataset = testset.get_dataset()

    for model_path in lst_models:
        path_tokens = model_path.split('/')
        exp_id = path_tokens[-2]

        if exp_id in tqdm(lst_exp):
            print('*' * 30)
            print(exp_id)
            print('*' * 30)

            # Load the model
            if 'ex' in exp_id:
                model = resnet18mam(NUM_CLASSES).to(device)
            elif 'code_lm' in exp_id:
                llm_block = 'code_lm'
                model = resnet18mamllm(NUM_CLASSES, llm_block=llm_block).to(device)
            elif 'large' in exp_id:
                llm_block = 'sent_transf_large'
                model = resnet18mamllm(NUM_CLASSES, llm_block=llm_block).to(device)
            elif 'random' in exp_id:
                llm_block = 'sent_transf'
                model = resnet18mamllm(NUM_CLASSES, llm_block=llm_block, llm_pretrain='False').to(device)
            else:
                model = resnet18mam(NUM_CLASSES).to(device)

            state_dict = torch.load(model_path)['state_dict']
            model.load_state_dict(state_dict) #, strict=False)
            model = model.cuda()

            results['id'].append(exp_id)
            results['alpha'].append(alpha)

            # Validate on Stylized TinyImageNet
            print('=' * 30)
            print(f'Validating on {dataset_name} with alpha={alpha}')

            acc = validate(model, test_dataset, True)
            print(f'Accuracy: {acc}')
            results['accuracy'].append(acc)

        # Save results to CSV for each alpha iteration
        df = pd.DataFrame(results)
        df.to_csv(f'/volumes1/vlm-cl/paper/stylized_tiny_llm.csv')

print('Validation on Stylized TinyImageNet Complete!')
