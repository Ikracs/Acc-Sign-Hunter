
import timm
import torch
import torchvision.models as tv_models
from torchvision import transforms
from abc import ABCMeta, abstractmethod

from datasets import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

SUPPORT_MODELS = [
    'resnet50', 'vgg16', 'inception_v3',
    'resnet50d', 'resnet101d', 'resnet152d',
    'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7',
    'deit_b', 'deit_l', 'deit_h',
    'swin_s', 'swin_b', 'swin_l',
    'vit_b', 'vit_l'
]


MODEL_ROOT = ''


class ClassificationModel(metaclass=ABCMeta):
    def __init__(self):
        self._load_model()
        self._init_model()

    @abstractmethod
    def _load_model(self):
        pass

    def _init_model(self):
        self.device = torch.device('cpu')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    @abstractmethod
    def _preprocess(self, x) -> torch.tensor:
        pass

    def __call__(self, x):
        return self.model(self._preprocess(x))


class ImageNet1kModel(ClassificationModel):

    DATASET = 'imagenet-1k'

    def _preprocess(self, x):
        return transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )(x)


# for reproduction of the performance in paper
class ResNet50(ImageNet1kModel):
    def _load_model(self):
        self.model = tv_models.resnet50(pretrained=True)


class VGG16(ImageNet1kModel):
    def _load_model(self):
        self.model = tv_models.vgg16_bn(pretrained=True)


# additional target models provided by timm
class InceptionV3(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('inception_v3', pretrained=True)


class ResNet50D(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('resnet50d', pretrained=True)


class ResNet101D(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('resnet101d', pretrained=True)


class ResNet152D(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('resnet152d', pretrained=True)


class EfficientNetB3(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('efficientnet_b3', pretrained=True)


class EfficientNetB4(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('efficientnet_b4', pretrained=True)


class EfficientNetB5(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('efficientnet_b5', pretrained=True)


class EfficientNetB6(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('efficientnet_b6', pretrained=True)


class EfficientNetB7(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('efficientnet_b7', pretrained=True)


class BaseDeiT(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('deit3_base_patch16_224', pretrained=True)


class LargeDeiT(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('deit3_large_patch16_224', pretrained=True)


class HugeDeiT(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('deit3_huge_patch14_224', pretrained=True)


class SmallSwin(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)


class BaseSwin(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)


class LargeSwin(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)


class BaseViT(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)


class LargeViT(ImageNet1kModel):
    def _load_model(self):
        self.model = timm.create_model('vit_large_patch16_224', pretrained=True)


def build_model(config):
    if config['model'] == 'resnet50':
        return ResNet50()
    elif config['model'] == 'vgg16':
        return VGG16()
    elif config['model'] == 'inception_v3':
        return InceptionV3()
    elif config['model'] == 'resnet50d':
        return ResNet50D()
    elif config['model'] == 'resnet101d':
        return ResNet101D()
    elif config['model'] == 'resnet152d':
        return ResNet152D()
    elif config['model'] == 'efficientnet_b3':
        return EfficientNetB3()
    elif config['model'] == 'efficientnet_b4':
        return EfficientNetB4()
    elif config['model'] == 'efficientnet_b5':
        return EfficientNetB5()
    elif config['model'] == 'efficientnet_b6':
        return EfficientNetB6()
    elif config['model'] == 'efficientnet_b7':
        return EfficientNetB7()
    elif config['model'] == 'deit_b':
        return BaseDeiT()
    elif config['model'] == 'deit_l':
        return LargeDeiT()
    elif config['model'] == 'deit_h':
        return HugeDeiT()
    elif config['model'] == 'swin_s':
        return SmallSwin()
    elif config['model'] == 'swin_b':
        return BaseSwin()
    elif config['model'] == 'swin_l':
        return LargeSwin()
    elif config['model'] == 'vit_b':
        return BaseViT()
    elif config['model'] == 'vit_l':
        return LargeViT()
    else:
        raise NotImplementedError
