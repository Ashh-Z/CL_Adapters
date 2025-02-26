import os
import pandas as pd
import torch
from glob import glob
import numpy as np
from norm_datasets.dataset import DATASETS
from backbone.ResNet_mam_llm import resnet18mamllm
from backbone.ResNet_mam import resnet18mam
from torchvision import transforms
from imagecorruptions import get_corruption_names, corrupt
from PIL import Image
from torch.autograd import Variable


def validate(model, dataset, cuda=False, verbose=True):
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
class ImageCorruptions:
    def __init__(self, severity, corruption_name):
        self.severity = severity
        self.corruption_name = corruption_name

    def __call__(self, image, labels=None):
        image = np.array(image)
        cor_image = corrupt(image, corruption_name=self.corruption_name,severity=self.severity)
        return Image.fromarray(cor_image)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

corrupt_list = ['brightness', 'contrast', 'fog', 'frost', 'snow',
                'gaussian_noise', 'shot_noise', 'impulse_noise',
                'motion_blur', 'defocus_blur', 'glass_blur', 'zoom_blur', 'gaussian_blur',
                'pixelate', 'elastic_transform', 'jpeg_compression', 'speckle_noise', 'spatter', 'saturate']

class arg_class():
    def __init__(self, dataset, model_architecture, num_classes):
        self.dataset = dataset
        self.model_architecture = model_architecture
        self.num_classes = num_classes

lst_models = glob(r'/volumes1/vlm-cl/normal_cls/*/model.ph')
lst_models += glob(r'/volumes1/vlm-cl/dytox_cls/implicit/*/model.ph')
lst_exp = [
    'normal-resnet18mam-cifar10-desc--e-100-s-42',
    'revproj-vlm-resnet18mam-cifar10-desc-e-100-l-sim-20.0-text-sent_transf-s-42',
    'normal-resnet18mamllm-cifar10-desc--l0.001-e-150-s-42'
    # 'vlm-resnet18mam-celeba-desc-e-100-l-0.05-26.0-text-sent_transf-s-42'
]

results = {
    'id': [],
    'corruption': [],
    'severity': [],
    'accuracy': [],
}

count = 0
severity = [3]

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
        dataset = DATASETS[data[0]](data[2])
        if 'llm' in exp_id:
            model = resnet18mamllm(dataset.NUM_CLASSES).to(device)
        else:
            model = resnet18mam(dataset.NUM_CLASSES).to(device)
        state_dict = torch.load(model_path)['state_dict']
        model.load_state_dict(state_dict)
        model = model.cuda()

        for corrupt_name in corrupt_list:
            for sev in severity:
                transform_test = transforms.Compose(
                    [
                        ImageCorruptions(sev, corrupt_name),
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=dataset.MEAN, std=dataset.STD)
                    ]
                )
                testset = dataset.get_dataset('test', None, transform_test)
                acc = validate(model, testset, True)

                results['id'].append(exp_id)
                results['corruption'].append(corrupt_name)
                results['severity'].append(sev)
                results['accuracy'].append(acc)

        count+=1
        df = pd.DataFrame(results)
        df.to_csv('/volumes1/vlm-cl/paper/corrupt_eval_nonorm_s5.csv')

print(count)