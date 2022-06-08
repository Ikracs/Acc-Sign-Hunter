import os
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from imagenet_labels import label_to_name


class Logger(object):
    def __init__(self):
        self.info = []

    def log_info(self, *args):
        self.info.append(list(args))

    def save(self, logging_pth):
        np.save(logging_pth, self.info)

def save_imgs(imgs, labels, save_root, budget=100):
    for count, (img, label) in enumerate(zip(imgs, labels)):
        if count >= budget: break
        save_name = '{:s}.jpg'.format(label_to_name(label))
        save_img(img, os.path.join(save_root, save_name))

def save_img(array, save_pth):
    image = array.transpose(1, 2, 0)
    plt.figure(); plt.imshow(image)
    plt.xticks([]); plt.yticks([]); plt.axis('off')
    plt.savefig(save_pth, bbox_inches='tight', pad_inches=0)
    plt.close()

def random_pseudo_label(label, n_cls=1000):
    target = np.zeros_like(label).long()
    for i_img in range(label.shape[0]):
        classes = list(range(n_cls)).remove(label[i_img])
        target[i_img] = np.random.choice(classes)
    return target

def get_log_name(config):
    save_name = config['model'].upper()
    save_name += '-attack={}'.format(config['attack'])
    save_name += '-loss={}'.format(config['loss'])
    save_name += '-targeted={}'.format(config['targeted'])
    save_name += '-budget={}'.format(config['budget'])
    save_name += '-epsilon={}'.format(config['epsilon'])
    save_name += '-n_ex={:d}'.format(config['n_ex'])
    return save_name + '.log'

