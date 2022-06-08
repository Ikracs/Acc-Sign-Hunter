import math
import torch
import numpy as np
from torch.nn import DataParallel
from torchvision.models import resnet50 as resnet
from torchvision.models import vgg16_bn as vgg
from torchvision.models import inception_v3 as inception


class Model:
    def __init__(self, config):
        self.loss_type  = config['loss']
        self.targeted   = config['targeted']
        self.batch_size = config['batch_size']

        model = eval(config['model'])(pretrained=True)
        self.model = DataParallel(model.cuda()).eval()

        m = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        s = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        self.M, self.S = m.astype(np.float32), s.astype(np.float32)

    def predict(self, x):
        x = ((x - self.M) / self.S).astype(np.float32)

        bs = self.batch_size
        bn = math.ceil(x.shape[0] / bs)
        
        logits_all = []
        with torch.no_grad():
            for i in range(bn):
                x_batch = x[i * bs: (i + 1) * bs]
                x_batch = torch.from_numpy(x_batch).cuda()
                logits = self.model(x_batch).cpu().numpy()
                logits_all.append(logits)
        logits_all = np.vstack(logits_all)
        return logits_all
    
    def loss(self, logits, label):
        one_hot_label = np.zeros_like(logits, dtype=np.bool)
        one_hot_label[np.arange(logits.shape[0]), label] = True
        
        if self.loss_type == 'cw':    # negative CW loss
            diff = logits - logits[one_hot_label][:, np.newaxis]
            diff[one_hot_label] = -np.inf
            loss = -diff.max(axis=1) if self.targeted else diff.max(axis=1)
        elif self.loss_type == 'ce':  # Cross-Entropy Loss
            # avoid overflow of 'exp' operation
            logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs  = logits / logits.sum(axis=1, keepdims=True)
            loss = -np.log(probs[one_hot_label])
            loss = -loss if self.targeted else loss
        else:
            raise NotImplementedError('Unknown loss: ' + self.loss_type)
        return loss
    
    def done(self, logits, label):
        if self.targeted:
            return logits.argmax(axis=-1) == label
        else:
            return logits.argmax(axis=-1) != label
