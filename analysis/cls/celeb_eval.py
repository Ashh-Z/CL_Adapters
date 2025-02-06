import pandas as pd
import torch
import numpy as np
from glob import glob
from norm_datasets.dataset import DATASETS
from backbone.ResNet_mam_llm import resnet18mamllm
from backbone.ResNet_mam import resnet18mam

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def eval_model(model, data_loader):
    y = []
    y_pred = []
    is_blond = []

    model.eval()

    for data, label, blond in data_loader:
        data, label, blond = data.cuda(), label.cuda(), blond.cuda()

        scores = model(data)
        scores = scores[0] if isinstance(scores, tuple) else scores
        _, predicted = scores.max(1)

        y.append(label.cpu())
        y_pred.append(predicted.detach().cpu())
        is_blond.append(blond.cpu())

    y = torch.cat(y).numpy()
    y_pred = torch.cat(y_pred).numpy()
    is_blond = torch.cat(is_blond).numpy()

    np.mean(y == y_pred)

    # Men Blonde
    num_samples = len(y)
    blonde_male = [(y[i] == 1) and is_blond[i] for i in range(num_samples)]
    blonde_female = [(y[i] == 0) and is_blond[i] for i in range(num_samples)]
    non_blonde_male = [(y[i] == 1) and not is_blond[i] for i in range(num_samples)]
    non_blonde_female = [(y[i] == 0) and not is_blond[i] for i in range(num_samples)]

    print('Overall:',  np.mean(y == y_pred))
    print('Blonde Male:',  np.mean(y[blonde_male] == y_pred[blonde_male]))
    print('Non Blonde Male:',  np.mean(y[non_blonde_male] == y_pred[non_blonde_male]))
    print('Blonde Female:',  np.mean(y[blonde_female] == y_pred[blonde_female]))
    print('Non Blonde Female:',  np.mean(y[non_blonde_female] == y_pred[non_blonde_female]))

    overall = np.mean(y == y_pred)
    blond_male = np.mean(y[blonde_male] == y_pred[blonde_male])
    nonblonde_male = np.mean(y[non_blonde_male] == y_pred[non_blonde_male])
    blond_female = np.mean(y[blonde_female] == y_pred[blonde_female])
    nonblond_female = np.mean(y[non_blonde_female] == y_pred[non_blonde_female])

    return overall, blond_male, nonblonde_male, blond_female, nonblond_female

class arg_class():
    def __init__(self, dataset, model_architecture, num_classes):
        self.dataset = dataset
        self.model_architecture = model_architecture
        self.num_classes = num_classes


lst_models = glob(r'/volumes1/vlm-cl/normal_cls/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/dytox_cls/*/model.ph')
lst_models +=glob(r'/volumes1/vlm-cl/seeds/cls/*/model.ph')
lst_exp = [
    # 'normal-resnet18mam-celeba-desc--e-100-s-42',
    # 'normal-resnet18mam-celeba-desc--e-100-s-0',
    # 'normal-resnet18mam-celeba-desc--e-100-s-1'
    'vlm-resnet18mam-celeba-desc-e-100-l-0.05-18.0-text-sent_transf-s-42',
    "vlm-resnet18mam-celeba-desc-e-100-l-0.05-26.0-text-sent_transf-s-42"
    # 'normal-resnet18mamllm-celeba-desc--l0.001-e-100-s-42',
    # 'ix-normal-resnet18mamllm-sent_transf-celeba-desc--l0.0001-e-100-s-43',
    # 'ix-normal-resnet18mamllm-sent_transf-celeba-desc--l0.0001-e-100-s-44'
]

results = {
    'id': [],
    'overall': [],
    'blond_m': [],
    'nonblond_m': [],
    'blond_f': [],
    'nonblond_f': [],
}

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
        data = ('celeba', 2, '/volumes1/datasets/celeba')
        dataset = DATASETS[data[0]](data[2])
        testset = dataset.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=4
        )
        if 'llm' in exp_id:
            model = resnet18mamllm(dataset.NUM_CLASSES, 64, 'sent_transf').to(device)
        else:
            model = resnet18mam(dataset.NUM_CLASSES).to(device)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict)
        model = model.cuda()

        overall, blond_male, nonblonde_male, blond_female, nonblond_female = eval_model(model, test_loader)

        results['id'].append(exp_id)
        results['overall'].append(overall)
        results['blond_m'].append(blond_male)
        results['nonblond_m'].append(nonblonde_male)
        results['blond_f'].append(blond_female)
        results['nonblond_f'].append(nonblond_female)

        count+=1
        df = pd.DataFrame(results)
        df.to_csv('/volumes1/vlm-cl/paper/celeb_eval_all.csv')

print(count)