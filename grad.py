import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages

import utils
from models import SUPPORT_MODELS, build_model
from datasets import SUPPORT_DATASETS, build_dataset
from losses import SUPPORT_LOSSES, build_criterion


def _calculate_local_sim(grad, scope, metric):
    H, W = grad.shape[1:]
    window_size = 2 * scope + 1
    sim_mat = np.zeros((window_size, window_size))
    for h in range(scope, H - scope):
        for w in range(scope, W - scope):
            cgrad = grad[:, h: h + 1, w: w + 1]
            beg_h, end_h = h - scope, h + scope + 1
            beg_w, end_w = w - scope, w + scope + 1
            wgrad = grad[:, beg_h: end_h, beg_w: end_w]
            if metric == 'dot':
                sim_mat += (cgrad * wgrad).sum(axis=0)
            elif metric == 'norm':
                sim_mat += np.linalg.norm(cgrad * wgrad, dim=0)
            else:
                raise NotImplementedError

    count = (H - 2 * scope) * (W - 2 * scope)
    return sim_mat / count


def _plot_sim_mat(sim_mat, save_pth, as_pdf=False):
    if as_pdf:
        save_pth = '.'.join([*save_pth.split('.'), 'pdf'])
        pdf = PdfPages(save_pth)

    plt.title('Local Similarity of Gradient')
    sim_mat -= sim_mat.min(); sim_mat /= sim_mat.max()
    plt.imshow(sim_mat, interpolation='nearest', cmap=plt.cm.jet)

    H, W = sim_mat.shape
    xs = np.arange(W) - (W - 1) // 2
    ys = np.arange(H) - (H - 1) // 2
    plt.xticks(np.arange(W), xs)
    plt.yticks(np.arange(H), ys)
    plt.xlabel('W')
    plt.ylabel('H')
    plt.colorbar()
    if as_pdf:
        pdf.savefig()
        pdf.close()
    else:
        plt.savefig(save_pth)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test local similarity of Gradient')
    parser.add_argument('--loss', type=str, choices=SUPPORT_LOSSES, help='adv loss function')
    parser.add_argument('--dataset', type=str, choices=SUPPORT_DATASETS, help='validation dataset')
    parser.add_argument('--model', type=str, choices=SUPPORT_MODELS, help='validation model')
    parser.add_argument('--sign', action='store_true', help='use grad sign or not')
    parser.add_argument('--n_ex', type=int, default=10, help='total num of imgs for test')
    parser.add_argument('--batch_size', type=int, default=10, help='num of imgs in a batch')
    parser.add_argument('--scope', type=int, default=3, help='scope for calculating sim')
    parser.add_argument('--metric', type=str, choices=['dot', 'norm'], help='sim metric')
    parser.add_argument('--gpu', type=str, default=None, help='available GPU id')
    parser.add_argument('--save_root', type=str, default='figs', help='save root of visualizations')
    parser.add_argument('--pdf', action='store_true', help='save as pdf file or not')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    cfg = vars(parser.parse_args())

    for k, v in cfg.items(): print(k, v)

    utils.set_random_seed(cfg['seed'])

    if cfg['gpu']:
        torch.cuda.manual_seed(cfg['seed'])
        device = torch.device('cuda:{}'.format(cfg['gpu']))
    else:
        device = torch.device('cpu')

    vmodel = build_model(cfg)
    criterion = build_criterion(cfg)
    dataset, _ = build_dataset(cfg)
    data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    save_root = os.path.join(cfg['save_root'], 'grad')
    if not os.path.exists(save_root): os.mkdir(save_root)

    vmodel.to(device)
    for b, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        images.requires_grad = True
        loss = criterion(vmodel(images), labels)
        loss.sum().backward()
        grads = images.grad.sign() if cfg['sign'] else images.grad

        for i, grad in enumerate(grads.cpu().numpy()):
            sim_mat = _calculate_local_sim(grad, cfg['scope'], cfg['metric'])

            idx = b * cfg['batch_size'] + i
            save_name = '{:d}#{:s}'.format(idx, 'sign' if cfg['sign'] else 'grad')
            _plot_sim_mat(sim_mat, os.path.join(save_root, save_name), cfg['pdf'])
            print("Visualization of img {:d} is saved in '{:s}'.".format(idx, save_root))
