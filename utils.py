import time
import json
import random
import platform
import collections

import torch
import numpy as np
from PIL import Image
from typing import List
from datetime import datetime


class Timer(object):
    def start(self, formatting=True):
        self.formatting = formatting
        self.start_time = time.time()

    def consume(self):
        elapsed = int(time.time() - self.start_time)
        if not self.formatting:
            return elapsed
        else:
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = (elapsed % 3600) % 60
            return hours, minutes, seconds


class Logger(object):
    def __init__(self):
        self.info = collections.defaultdict(list)

    def log_info(self, info):
        for k, v in info.items():
            self.info[k] += [v]

    def save_log(self, save_name):
        info = {k: (v if len(v) > 1 else v[0]) for k, v in self.info.items()}
        with open(save_name, 'w') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)


def get_time_stamp():
    time_stamp = '-'.join(str(datetime.now())[:-7].split(' '))
    if platform.system() == 'Windows':
        time_stamp = time_stamp.replace(':', '_')
    return time_stamp


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def printf(info, end='\n'):
    for k, v in info.items():
        if isinstance(v, float):
            print("{:s}: {:.3f}".format(k, v), end=' ')
        else:
            print("{:s}: {}".format(k, v), end=' ')
    print(end=end)


def save_img(img, save_pth):
    img = img.transpose(1, 2, 0)
    img = np.uint8(255 * img)
    img = Image.fromarray(img)
    img.save(save_pth)


def id_to_onehot(num_classes, id):
    one_hot = torch.zeros(id.shape[0], num_classes).to(id.device)
    return one_hot.scatter(-1, id.unsqueeze(-1), 1).bool()


def random_pseudo_label(label, n_cls):
    target = torch.zeros_like(label)
    for i in range(label.shape[0]):
        classes = list(range(n_cls))
        classes.remove(label[i])
        target[i] = random.choice(classes)
    return target


def _pad_sequences(seqs, mode='replicate', value=0):
    if mode == 'constant':
        max_len = max([len(s) for s in seqs])
        seqs = [s + [value] * (max_len - len(s)) for s in seqs]
    elif mode == 'replicate':
        max_len = max([len(s) for s in seqs])
        seqs = [s + [s[-1]] * (max_len - len(s)) for s in seqs]
    else:
        raise NotImplementedError
    return seqs


def squeeze_info_over_batches(v: List [List]):
    padded = _pad_sequences(v)
    v = torch.tensor(padded).float().mean(dim=0)
    return v.numpy().tolist()


def concat_info_over_batches(v: List [List]):
    return [v[0].extend(l) for l in v[1:]]


def calculate_adv_accuracy(accuracy, asr):
    return accuracy * (1 - asr)


def calculate_cos_sim(t1, t2, reduction=True):
    assert t1.shape == t2.shape, "Tha shape of t1 and t2 are dismatched!"
    _flatten = lambda x: x.view(x.shape[0], -1)
    _normalize = lambda x: x / x.norm(p=2, dim=-1, keepdim=True)

    t1 = _normalize(_flatten(t1))
    t2 = _normalize(_flatten(t2))
    cos_sim = (t1 * t2).sum(dim=-1)
    return cos_sim.mean() if reduction else cos_sim
