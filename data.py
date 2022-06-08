import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_ROOT = '/mnt/datadev_2/std/lisiyuan/imagenet'

IMAGENET_TRANS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

INCEPTION_TRANS = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor()
])


class ImageNet(Dataset):
    def __init__(self, trans=None, idxs=None):
        self.trans, self.idxs = trans, idxs
        droot = os.path.join(IMAGENET_ROOT, 'val')
        self.imgs = sorted([os.path.join(droot, i) for i in os.listdir(droot)])

        label_path = os.path.join(IMAGENET_ROOT, 'val.txt')
        with open(label_path) as label_file:
            labels = [i.split(' ') for i in label_file.read().strip().split('\n')]
            self.labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    def _load_image(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def __getitem__(self, index):
        if self.idxs is not None:
            index = self.idxs[index]
        image = self._load_image(self.imgs[index])
        label = self.labels[os.path.basename(self.imgs[index])]
        
        if self.trans is not None:
            image = self.trans(image)
        return image, label

    def __len__(self):
        return len(self.imgs) if self.idxs is None else len(self.idxs)

def load_imagenet(n_ex, trans, idxs=None):
    loader = DataLoader(
        ImageNet(trans, idxs),
        batch_size=n_ex, shuffle=False,
        num_workers=3
    )
    x_test, y_test = next(iter(loader))
    return np.array(x_test, dtype=np.float32), np.array(y_test)
