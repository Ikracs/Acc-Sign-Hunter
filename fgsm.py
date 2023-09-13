import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

import utils
from attacks import NoisyFGSM, NoisyIFGSM
from models import SUPPORT_MODELS, build_model
from datasets import SUPPORT_DATASETS, build_dataset
from losses import SUPPORT_LOSSES, build_criterion


if __name__ == '__main__':

    time_stamp = utils.get_time_stamp()

    parser = argparse.ArgumentParser(description='Noisy FGSM/I-FGSM against classification')
    parser.add_argument('--attack', type=str, choices=['fgsm', 'ifgsm'], help='attack method')
    parser.add_argument('--config', type=str, default=None, help='cfg file pth for saving attack params')
    parser.add_argument('--loss', type=str, choices=SUPPORT_LOSSES, help='adv loss function')
    parser.add_argument('--dataset', type=str, choices=SUPPORT_DATASETS, help='validation dataset')
    parser.add_argument('--model', type=str, choices=SUPPORT_MODELS, help='victim model for attack')
    parser.add_argument('--targeted', action='store_true', help='targeted attack or not')
    parser.add_argument('--n_ex', type=int, default=1000, help='total num of imgs to attack')
    parser.add_argument('--batch_size', type=int, default=32, help='num of imgs attacked once')
    parser.add_argument('--gpu', type=str, default=None, help='available GPU id')
    parser.add_argument('--log_root', type=str, default=None, help='log root of attacking')
    parser.add_argument('--verbose', action='store_true', help='print per-batch results to console or not')
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
            print("Error when loading params from '{:s}'.".format(cfg['config']))
    else:
        print('No assigned cfg. Use default params.')

    if cfg['gpu']:
        torch.cuda.manual_seed(cfg['seed'])
        device = torch.device('cuda:{}'.format(cfg['gpu']))
    else:
        device = torch.device('cpu')

    vmodel = build_model(cfg)
    dataset, _ = build_dataset(cfg)
    data_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    timer, logger = utils.Timer(), utils.Logger()

    find_corr = lambda x, y: x.argmax(dim=-1) == y
    if cfg['targeted']:
        find_done = lambda x, y: find_corr(x, y)
    else:
        find_done = lambda x, y: ~find_corr(x, y)

    attacker = NoisyFGSM(cfg) if cfg['attack'] == 'fgsm' else NoisyIFGSM(cfg)
    criterion = build_criterion(cfg)

    attacker.set_victim_model(vmodel)
    attacker.set_criterion(criterion)
    attacker.set_done_cond(find_done)

    timer, logger = utils.Timer(), utils.Logger()

    avg_accuracy = 0.0

    timer.start()
    vmodel.to(device)
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        logits = vmodel(images)
        cidx = torch.where(find_corr(logits, labels))[0]
        accuracy = cidx.shape[0] / images.shape[0]
        avg_accuracy += accuracy

        if cfg['verbose']:
            print('Clean Accuracy in batch {:d}: {:.3f}'.format(i, accuracy))
            print('Start attacking correctly classified images...')

        if cfg['targeted']:
            labels = utils.random_pseudo_label(labels, dataset.NUM_CLASSES)

        logger.log_info(attacker.run((images[cidx], labels[cidx])))

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
