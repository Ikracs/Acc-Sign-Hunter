import argparse

import torch
from torch.utils.data import DataLoader

import utils
from models import SUPPORT_MODELS, build_model
from datasets import SUPPORT_DATASETS, build_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test accuracy of classification models')
    parser.add_argument('--dataset', type=str, choices=SUPPORT_DATASETS, help='validation dataset')
    parser.add_argument('--n_ex', type=int, default=1000, help='total num of imgs for test')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for infer')
    parser.add_argument('--gpu', type=str, default=None, help='available GPU id')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    cfg = vars(parser.parse_args())

    utils.set_random_seed(cfg['seed'])

    if cfg['gpu']:
        gpu = cfg['gpu'].split(',')[0]
        device = torch.device('cuda:{:s}'.format(gpu))
    else:
        device = torch.device('cpu')

    find_corr = lambda x, y: x.argmax(dim=-1) == y

    timer = utils.Timer()
    dataset, _ = build_dataset(cfg)
    data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    for model_type in SUPPORT_MODELS:
        model = build_model({**cfg, 'model': model_type}).to(device)

        if model.DATASET != cfg['dataset']: continue

        timer.start()
        avg_accuracy = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            cidx = torch.where(find_corr(model(images), labels))[0]
            avg_accuracy += cidx.shape[0] / images.shape[0]
        avg_accuracy /= len(data_loader)

        hours, minutes, seconds = timer.consume()
        print('Clean Accuracy of {:s}: {:.3f},'.format(model_type, avg_accuracy), end=' ')
        print('Inference Time: {:d} h {:d} m {:d} s'.format(hours, minutes, seconds))
