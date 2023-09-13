import torch
import torch.nn.functional as F

import utils

SUPPORT_LOSSES = ['cw', 'ce']


class CWLoss(torch.nn.Module):
    def __init__(self, config):
        super(CWLoss, self).__init__()
        self.targeted = config.get('targeted', False)

    def forward(self, x, y):
        y = utils.id_to_onehot(x.shape[-1], y)
        diff = x - x[y][..., None]
        diff[y] = -torch.inf
        loss = diff.max(dim=-1)[0]
        return -loss if self.targeted else loss


class CELoss(torch.nn.Module):
    def __init__(self, config):
        super(CELoss, self).__init__()
        self.targeted = config.get('targeted', False)

    def forward(self, x, y):
        loss = F.cross_entropy(x, y, reduction='none')
        return -loss if self.targeted else loss


def build_criterion(config):
    if config['loss'] == 'cw':
        return CWLoss(config)
    elif config['loss'] == 'ce':
        return CELoss(config)
    else:
        raise NotImplementedError
