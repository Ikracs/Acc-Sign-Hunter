import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

import utils
from models import SUPPORT_MODELS, build_model
from datasets import SUPPORT_DATASETS, build_dataset
from attacks import SUPPORT_ATTACKS, build_attacker
from losses import SUPPORT_LOSSES, build_criterion


if __name__ == '__main__':

    time_stamp = utils.get_time_stamp()

    parser = argparse.ArgumentParser(description='Black-box attack against classification')
    parser.add_argument('--attack', type=str, choices=SUPPORT_ATTACKS, help='attack method')
    parser.add_argument('--config', type=str, default=None, help='cfg file pth for saving attack params')
    parser.add_argument('--loss', type=str, choices=SUPPORT_LOSSES, help='adv loss function')
    parser.add_argument('--dataset', type=str, choices=SUPPORT_DATASETS, help='validation dataset')
    parser.add_argument('--model', type=str, choices=SUPPORT_MODELS, help='victim model')
    parser.add_argument('--targeted', action='store_true', help='targeted attack or not')
    parser.add_argument('--budget', type=int, default=10000, help='query budget for black-box attack')
    parser.add_argument('--n_ex', type=int, default=1000, help='total num of imgs to attack')
    parser.add_argument('--batch_size', type=int, default=64, help='num of imgs attacked once')
    parser.add_argument('--gpu', type=str, default=None, help='available GPU id')
    parser.add_argument('--log_root', type=str, default=None, help='log root of attacking')
    parser.add_argument('--save_root', type=str, default=None, help='save root of adv imgs')
    parser.add_argument('--verbose', action='store_true', help='print per-batch results to console or not')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency if verbose')
    parser.add_argument('--early_stop', action='store_true', help='early-stop in black-box attack or not')
    parser.add_argument('--DEBUG', action='store_true', help='use DEBUG mode or not')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    cfg = vars(parser.parse_args())

    for k, v in cfg.items(): print(k, v)

    utils.set_random_seed(cfg['seed'])

    if cfg['config'] is not None:
        try:
            with open(cfg['config'], 'r') as f:
                cfg.update(json.load(f))
                print("Successfully load params from '{:s}'.".format(cfg['config']))
        except Exception as e:
            print(e)
            print("An error occured when loading params from '{:s}'.".format(cfg['config']))
            print('Use default params.')
    else:
        print('No assigned cfg. Use default params.')

    if cfg['gpu']:
        gpu = cfg['gpu'].split(',')[0]
        device = torch.device('cuda:{:s}'.format(gpu))
    else:
        device = torch.device('cpu')

    vmodel = build_model(cfg)
    dataset, label_to_name = build_dataset(cfg)
    data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    attacker = build_attacker(cfg)
    criterion = build_criterion(cfg)

    find_corr = lambda x, y: x.argmax(dim=-1) == y
    if not cfg['early_stop']:
        find_done = lambda x, y: torch.zeros(x.shape[0]).bool().to(x.device)
    elif cfg['targeted']:
        find_done = lambda x, y: find_corr(x, y)
    else:
        find_done = lambda x, y: ~find_corr(x, y)

    attacker.set_victim_model(vmodel)
    attacker.set_criterion(criterion)
    attacker.set_done_cond(find_done)

    timer, logger = utils.Timer(), utils.Logger()

    if cfg['DEBUG']:
        print('Run in DEBUG mode:')

    if cfg['save_root']:
        ori_root = os.path.join(cfg['save_root'], 'ori')
        adv_root = os.path.join(cfg['save_root'], 'adv')
        if not os.path.exists(ori_root): os.mkdir(ori_root)
        if not os.path.exists(adv_root): os.mkdir(adv_root)

    idx, avg_accuracy = 0, 0.0

    timer.start()
    vmodel.to(device)
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        logits = vmodel(images)
        cidx = torch.where(find_corr(logits, labels))[0]
        accuracy = cidx.shape[0] / images.shape[0]
        avg_accuracy += accuracy

        if cfg['save_root']:

            def save_imgs(x, y, adv_x, adv_y, done, **kwargs):
                for i in range(x.shape[0]):
                    if not cfg['early_stop'] or cfg['early_stop'] and done[i]:
                        ori_name = '{:d}#{:s}.jpg'.format(idx + i, label_to_name(y[i].item()))
                        adv_name = '{:d}#{:s}.jpg'.format(idx + i, label_to_name(adv_y[i].item()))
                        utils.save_img(x[i].cpu().numpy(), os.path.join(ori_root, ori_name))
                        utils.save_img(adv_x[i].cpu().numpy(), os.path.join(adv_root, adv_name))

            attacker.register_exit_hooks(save_imgs=save_imgs)

        if cfg['verbose']:
            print('Clean Accuracy in batch {:d}: {:.3f}'.format(i, accuracy))
            print('Start attacking correctly classified images...')

        if cfg['targeted']:
            labels = utils.random_pseudo_label(labels, dataset.NUM_CLASSES)

        logger.log_info(attacker.run((images[cidx], labels[cidx])))

        idx += images.shape[0]

        if cfg['verbose'] and cfg['save_root']:
            print("Images are saved in '{:s}'.".format(cfg['save_root']))

    for k, v in logger.info.items():
        logger.info[k] = utils.squeeze_info_over_batches(v)

    avg_accuracy /= len(data_loader)
    print('Clean Accuracy: {:.3f}'.format(avg_accuracy))
    print('Results over all batches:')
    utils.printf({k: v[-1] for k, v in logger.info.items()})

    if cfg['log_root']:
        log_name = time_stamp + '.json'
        save_pth = os.path.join(cfg['log_root'], log_name)

        logger.log_info({
            'loss'    : cfg['loss'],
            'dataset' : cfg['dataset'],
            'model'   : cfg['model'],
            'targeted': cfg['targeted']
        })
        logger.log_info(attacker.params)

        logger.save_log(save_pth)
        print("Log file is saved in '{:s}'.".format(save_pth))

    hours, minutes, seconds = timer.consume()
    print('Time Elapsed: {:d} h {:d} m {:d} s'.format(hours, minutes, seconds))
