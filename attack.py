import os
import sys
import time
import math
import torch
import numpy as np

import utils
from imagenet_labels import label_to_name

def sign_hunter(model, img, label, config):
    budget    = config['budget']
    epsilon   = config['epsilon']
    
    log_freq  = config['log_freq']
    log_root  = config['log_root']
    save_root = config['save_root']

    logger = utils.Logger()

    min_v, max_v = 0, 1 if img.max() <= 1 else 255
    
    dim = np.prod(img.shape[1:])
    sign_bits = np.ones((img.shape[0], dim))
    query = np.ones(img.shape[0])
    
    delta = epsilon * sign_bits.reshape(img.shape)
    perturbed = np.clip(img + delta, min_v, max_v)
    
    logits = model.predict(perturbed)
    loss = model.loss(logits, label)
    done = model.done(logits, label)

    node_i, tree_h = 0, 0
    rs = math.ceil(dim / (2 ** tree_h))

    start_time = time.time()
    print('Strat attacking correct classified images...')
    for i_iter in range(budget):
        if (rs < 1) or (done.sum() == img.shape[0]): break

        sign_bits_new = sign_bits.copy()[~done]
        sign_bits_new[:, node_i * rs: (node_i + 1) * rs] *= -1
        
        delta = epsilon * sign_bits_new.reshape(img[~done].shape)
        perturbed = np.clip(img[~done] + delta, min_v, max_v)

        logits = model.predict(perturbed)
        loss_new = model.loss(logits, label[~done])
        done_new = model.done(logits, label[~done])
        query[~done] += 1

        improved = (loss_new > loss[~done])
        loss[~done] = improved * loss_new + ~improved * loss[~done]
        sign_bits[~done] = improved[:, np.newaxis] * sign_bits_new + \
            ~improved[:, np.newaxis] * sign_bits[~done]
        done[~done] = done_new
        
        node_i += 1
        if node_i == 2 ** tree_h:
            node_i = 0; tree_h += 1
            rs = math.ceil(dim / (2 ** tree_h))

        if i_iter % log_freq == 0 and done.sum().item() > 0:
            print('[Iter  {:0>4d}] '.format(i_iter), end='')
            print('Success rate: {:.3f}, '.format(done.mean()), end='')
            print('Avg. query: {:.2f}, '.format(query[done].mean()), end='')
            print('Med. query: {:.0f}'.format(np.median(query[done])))
            logger.log_info(done.mean(), query[done].mean())
    
    if log_root is not None:
        save_name = utils.get_log_name(config)
        logger.save(os.path.join(log_root, save_name))
    
    if save_root is not None:
        ori_root = os.path.join(save_root, 'clean')
        utils.save_imgs(img, label, ori_root)

        adv_root = os.path.join(save_root, 'adv')
        delta = epsilon * sign_bits[done].reshape(img[done].shape)
        perturbed = np.clip(img[done] + delta, min_v, max_v)
        adv_label = model.predict(perturbed).argmax(axis=-1)
        utils.save_imgs(perturbed, adv_label, adv_root)
    
    print('Final results: ', end='')
    print('Success rate: {:.3f}, Avg. query: {:.2f}, Med. query: {:.0f}'.format(
        done.mean(), query[done].mean(), np.median(query[done])))
    
    elapsed = time.time() - start_time
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Time Elapsed: {:.0f} h {:.0f} m'.format(hours, minutes))

def accelerated_sign_hunter(model, imgs, labels, config):
    budget    = config['budget']
    log_freq  = config['log_freq']
    log_root  = config['log_root']
    save_root = config['save_root']

    logger = utils.Logger()
    
    total_num_correct = 0
    total_num_queries = []
    adv_imgs, adv_labels = [], []

    start_time = time.time()
    print('Start attacking correct classified images...')
    for count, (img, label) in enumerate(zip(imgs, labels)):
        img, label = img[np.newaxis, ...], label[np.newaxis, ...]
        done, adv_img, queries = _ash_single_img(model, img, label, config)
        adv_label = model.predict(adv_img).argmax(axis=-1).item()
        
        print('Attack on {:d}th img starts, '.format(count), end='')
        print('original class: {:s}. '.format(label_to_name(label.item())), end='')
        
        if done:    # attack succeeds
            total_num_correct += 1
            total_num_queries.append(queries)
            adv_imgs.append(adv_img)
            adv_labels.append(adv_label)
            
            print('Attack succeeds, ', end='')
            print('final class: {:s}, '.format(label_to_name(adv_label)), end='')
            print('num queries: {:d}'.format(queries))
        else:       # attack fails
            print('Attack fails, ', end='')
            print('final class: {:s}'.format(label_to_name(adv_label)))
    
    if log_root is not None:
        total_num_queries = np.array(total_num_queries)
        for i_iter in range(0, budget, log_freq):
            succeed = total_num_queries < i_iter
            if succeed.sum() != 0:
                logger.log_info(
                    succeed.sum() / imgs.shape[0],
                    total_num_queries[succeed].mean()
                )
            else: logger.log_info(0.0, 0)
        save_name = utils.get_log_name(config)
        logger.save(os.path.join(log_root, save_name))
    
    if save_root is not None:
        ori_root = os.path.join(save_root, 'clean')
        utils.save_imgs(imgs, labels, ori_root)
        
        adv_root = os.path.join(save_root, 'adv')
        utils.save_imgs(adv_imgs, adv_labels, adv_root)
    
    print('Final results: ', end='')
    print('Success rate: {:.3f}, Avg. query: {:.2f}, Med. query: {:.0f}'.\
        format(
            total_num_correct / imgs.shape[0],
            np.mean(total_num_queries),
            np.median(total_num_queries)
        )
    )
    
    elapsed = time.time() - start_time
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Time Elapsed: {:.0f} h {:.0f} m'.format(hours, minutes))


def _ash_single_img(model, img, label, config):
    budget    = config['budget']
    epsilon   = config['epsilon']
    
    min_v, max_v = 0, 1 if img.max() <= 1 else 255

    dim = np.prod(img.shape)
    sign_bits = np.ones(dim)
    num_queries = 1

    delta = epsilon * sign_bits.reshape(img.shape)
    perturbed = np.clip(img + delta, min_v, max_v)
    
    logits = model.predict(perturbed)
    loss = model.loss(logits, label)
    done = model.done(logits, label)

    node_i, tree_h = 0, 0
    regions = [[0.0, [0, dim]]]

    def _divide(regions):
        regions_new = []
        for region in regions:
            start, end = region[1]
            mid = start + (end - start) // 2
            regions_new.append([region[0], [start, mid]])
            regions_new.append([region[0], [mid, end]])
        return regions_new
    
    while num_queries < budget and not done:
        need_query = True
        if node_i % 2 == 1:
            regions[node_i][0] -= regions[node_i - 1][0]
            need_query = regions[node_i][0] < 0
        
        if need_query:
            sign_bits_new = sign_bits.copy()
            start, end = regions[node_i][1]
            sign_bits_new[start: end] *= -1

            if start != end:
                delta = epsilon * sign_bits_new.reshape(img.shape)
                perturbed = np.clip(img + delta, min_v, max_v)

                logits = model.predict(perturbed)
                loss_new = model.loss(logits, label)
                done = model.done(logits, label)
                num_queries += 1

                regions[node_i][0] = (loss - loss_new).item()

                if loss_new > loss:
                    loss, sign_bits = loss_new, sign_bits_new
            else:
                regions[node_i][0] = float('inf')

        node_i += 1
        if node_i == 2 ** tree_h:
            node_i = 0; tree_h += 1
            
            regions = [[abs(r[0]), r[1]] for r in _divide(regions)]
            regions = sorted(regions, key=lambda r: r[0], reverse=False)
    
    delta = epsilon * sign_bits.reshape(img.shape)
    adv_img = np.clip(img + delta, min_v, max_v)
    return done, adv_img, num_queries
