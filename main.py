import os
import sys
import time
import argparse

import torch
import numpy as np

import data
import utils
from model import Model
from attack import sign_hunter
from attack import accelerated_sign_hunter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Black-Box attack against image classification')
    parser.add_argument('--attack', type=str, default='ash', choices=['sh', 'ash'])
    parser.add_argument('--loss', type=str, default='cw', choices=['cw', 'ce'])
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vgg', 'inception'])
    parser.add_argument('--idxs', type=str, default=None, help='file for saving idxs of sampled imgs')
    parser.add_argument('--targeted', action='store_true', help='targeted attack or not')
    parser.add_argument('--budget', type=int, default=10000, help='query budget for black-box attack')
    parser.add_argument('--epsilon', type=float, default=0.05, help='maximum linf norm of perturbation')
    parser.add_argument('--n_ex', type=int, default=1000, help='total num of images to attack')
    parser.add_argument('--batch_size', type=int, default=32, help='num of images attacked in an iter')
    parser.add_argument('--gpu', type=str, default='0', help='Available GPU id')
    parser.add_argument('--log_freq', type=int, default=50, help='log frequency')
    parser.add_argument('--log_root', type=str, default=None, help='log root of attacking')
    parser.add_argument('--save_root', type=str, default=None, help='save root of adv images')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    
    cfg = vars(parser.parse_args())
    for key in cfg.keys():
        print(key + ' ' + str(cfg[key]))

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu']

    victim_model = Model(cfg)
    if cfg['model'] == 'inception':
        trans = data.IMAGENET_TRANS 
    else:
        trans = data.INCEPTION_TRANS
    if cfg['idxs'] is not None and os.path.exists(cfg['idxs']):
        img_idxs = np.load(cfg['idxs'])
        img, label = data.load_imagenet(cfg['n_ex'], trans, img_idxs)
    else:
        img, label = data.load_imagenet(cfg['n_ex'], trans)
    
    if cfg['targeted']:
        label = utils.random_pseudo_label(label, 1000)
    
    attack = sign_hunter if cfg['attack'] == 'sh' else accelerated_sign_hunter
    attack(victim_model, img, label, cfg)
