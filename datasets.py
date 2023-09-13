import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SUPPORT_DATASETS = ['imagenet-1k']

IMAGENET_ROOT = ''

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD  = [0.229, 0.224, 0.225]


class ImageNet1K(Dataset):

    NUM_CLASSES  = 1000

    def __init__(self, config):
        self.n_ex = config.get('n_ex', None)

        droot = os.path.join(IMAGENET_ROOT, 'val')
        self.imgs = sorted([os.path.join(droot, i) for i in os.listdir(droot)])

        label_path = os.path.join(IMAGENET_ROOT, 'val.txt')
        with open(label_path) as label_file:
            labels = [i.split(' ') for i in label_file.read().strip().split('\n')]
            self.labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    @property
    def transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def _load_img(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def __getitem__(self, index):
        img = self._load_img(self.imgs[index])
        label = self.labels[os.path.basename(self.imgs[index])]
        return self.transform(img), label

    def __len__(self):
        return self.n_ex if self.n_ex else len(self.imgs)


def build_dataset(config):
    if config['dataset'] == 'imagenet-1k':
        from imagenet1k_labels import label_to_name
        return ImageNet1K(config), label_to_name
    else:
        raise NotImplementedError
