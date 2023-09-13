import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod

import utils

SUPPORT_ATTACKS = ['sh', 'ash']


class Attack(metaclass=ABCMeta):
    """
    Base Attack Class.
    """
    def __init__(self, config):
        self.DEBUG   = config.get('DEBUG', False)
        self.verbose = config.get('verbose', False)
        self.P_FREQ  = config.get('print_freq', 1)

        self.vmodel    = None
        self.criterion = None
        self.find_done = None

        self.params = {'attack': self.__str__()}
        self._stats = defaultdict(list)

        self.reset_iter_hooks()
        self.reset_exit_hooks()

    def set_victim_model(self, vmodel):
        self.vmodel = vmodel

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_done_cond(self, find_done):
        self.find_done = find_done

    @abstractmethod
    def _attack_main(self, images, labels):
        """
        Main attack method.
        """
        pass

    def run(self, samples):
        assert self.vmodel is not None, "call 'set_victim_model' before start an attack!"
        assert self.criterion is not None, "call 'set_criterion' before start an attack!"

        if self.find_done is None:
            print('WARNING: No attack done condition is set.')
            self.find_done = lambda x, y: torch.zeros(x.shape[0]).bool().to(x.device)

        images, labels = samples
        if not images.shape[0] or not labels.shape[0]:
            return defaultdict(list)

        return self._attack_main(images, labels)

    def reset_iter_hooks(self):
        self.iter_hooks = {}

    def register_iter_hooks(self, *args, **kwargs):
        for hook in args:
            hname = len(self.iter_hooks)
            self.iter_hooks[hname] = hook
        for hname, hook in kwargs.items():
            self.iter_hooks[hname] = hook

    def delte_iter_hook(self, hname):
        del self.iter_hooks[hname]

    def reset_exit_hooks(self):
        self.exit_hooks = {}

    def register_exit_hooks(self, *args, **kwargs):
        for hook in args:
            hname = len(self.exit_hooks)
            self.exit_hooks[hname] = hook
        for hname, hook in kwargs.items():
            self.exit_hooks[hname] = hook

    def delte_exit_hook(self, hname):
        del self.exit_hooks[hname]

    def _update_stats(self, stats):
        assert isinstance(stats, dict)
        for k, v in stats.items():
            self._stats[k] += [v]

    def _call_iter_hooks(self, iter_i, iter_args):
        if iter_i % self.P_FREQ == 0:
            if self.verbose:
                print('[{:0>5d}]'.format(iter_i), end=' ')

            for hook in self.iter_hooks.values():
                stats = hook(**iter_args)
                if stats is not None:
                    self._update_stats(stats)
                    if self.verbose:
                        utils.printf(stats, end='')

            if self.verbose: print('')

    def _call_exit_hooks(self, exit_args):
        if self.verbose:
            print('Final Results:', end=' ')

        for hook in self.exit_hooks.values():
            stats = hook(**exit_args)
            if stats is not None:
                self._update_stats(stats)
                if self.verbose:
                    utils.printf(stats, end='')

        if self.verbose: print('')

        stats_all = self._stats
        # reset stats dict for an another attack
        self._stats = defaultdict(list)

        return stats_all

    @abstractmethod
    def __str__(self):
        pass


class WhiteBoxAttack(Attack):
    def __init__(self, config):
        super(WhiteBoxAttack, self).__init__(config)

        self.epsilon = config.get('epsilon', 0.05)

        self.params.update({'epsilon': self.epsilon})

        self.register_iter_hooks(self.evaluate)
        self.register_exit_hooks(self.evaluate)

    def evaluate(self, loss, done, **kwargs):
        stats = {}
        stats['ASR'] = done.float().mean().item()
        stats['Adv Loss'] = loss.mean().item()
        return stats


class NoisyFGSM(WhiteBoxAttack):
    def __init__(self, config):
        super(NoisyFGSM, self).__init__(config)

        self.noisy = config.get('noisy', 0.0)

        self.params.update({'noisy': self.noisy})

    def _perturb(self, sign, reverse=True):
        mask = torch.rand_like(sign) > self.noisy
        if reverse:
            sign = sign * mask - sign * ~mask
        else:
            noise = torch.randn_like(sign).sign()
            sign = sign * mask + noise * ~mask
        return sign

    def _craft_adv_examples(self, x, s):
        delta = self.epsilon * self._perturb(s)
        return torch.clamp(x + delta, 0.0, 1.0)

    def _attack_main(self, images, labels):
        images.requires_grad = True

        logits = self.vmodel(images)
        loss = self.criterion(logits, labels)
        loss.sum().backward()
        sign = images.grad.sign()

        perturbed = self._craft_adv_examples(images, sign)

        logits = self.vmodel(perturbed)
        loss = self.criterion(logits, labels)
        done = self.find_done(logits, labels)

        labels = self.vmodel(images).argmax(dim=-1)

        return self._call_exit_hooks({
            'x': images,
            'y': labels,
            'adv_x': perturbed,
            'adv_y': logits.argmax(dim=-1),
            'loss': loss,
            'done': done
        })

    def __str__(self):
        return 'Noisy Fast Gradient Sign Method'


class NoisyIFGSM(NoisyFGSM):
    def __init__(self, config):
        super(NoisyIFGSM, self).__init__(config)

        self.alpha  = config.get('alpha', 5e-3)
        self.gamma  = config.get('momentum', 0.5)
        self.iter_n = config.get('iter_n', 10)

        self.params.update({
            'alpha' : self.alpha,
            'gamma' : self.gamma,
            'iter_n': self.iter_n
        })

        if self.DEBUG:
            self.register_iter_hooks(self._cal_grad_cont)

    def _cal_grad_cont(self, grad, **kwargs):
        _flatten = lambda x: x.view(x.shape[0], -1)
        _normalize = lambda x: x / x.norm(dim=-1, keepdim=True)

        if not hasattr(self, 'buffer'):
            self.buffer = grad
        else:
            g1 = _normalize(_flatten(self.buffer))
            self.buffer = grad
            g2 = _normalize(_flatten(self.buffer))
            consistency = (g1 * g2).sum(dim=-1).mean()
            return {'Grad Cont': consistency.item()}

    def _craft_adv_examples(self, x, s, minv, maxv):
        delta = self.alpha * self._perturb(s)
        return torch.clamp(x + delta, minv, maxv)

    def _attack_main(self, images, labels):
        minv = torch.clamp(images - self.epsilon, min=0.0)
        maxv = torch.clamp(images + self.epsilon, max=1.0)

        perturbed = images.clone()
        delta = torch.zeros_like(perturbed)

        for iter_i in range(self.iter_n):
            perturbed.requires_grad = True

            logits = self.vmodel(perturbed)
            loss = self.criterion(logits, labels)
            done = self.find_done(logits, labels)
            loss.sum().backward()

            perturbed.requires_grad = False

            delta = delta * self.gamma + perturbed.grad
            perturbed = self._craft_adv_examples(perturbed, delta.sign(), minv, maxv)

            self._call_iter_hooks(iter_i, {
                'loss': loss,
                'done': done,
                'grad': delta
            })

        logits = self.vmodel(perturbed)
        loss = self.criterion(logits, labels)
        done = self.find_done(logits, labels)

        labels = self.vmodel(images).argmax(dim=-1)

        return self._call_exit_hooks({
            'x': images,
            'y': labels,
            'adv_x': perturbed,
            'adv_y': logits.argmax(dim=-1),
            'loss': loss,
            'done': done
        })

    def _call_exit_hooks(self, exit_args):
        stats_all = super(NoisyIFGSM, self).\
            _call_exit_hooks(exit_args)
        if self.DEBUG: del self.buffer
        return stats_all

    def __str__(self):
        return 'Noisy Iterative Fast Gradient Sign Method'


class BlackBoxAttack(Attack):
    def __init__(self, config):
        super(BlackBoxAttack, self).__init__(config)

        self.budget  = config.get('budget', 1000)

        self.params.update({'budget' : self.budget})


class ScoreBasedAttack(BlackBoxAttack):
    def __init__(self, config):
        super(ScoreBasedAttack, self).__init__(config)

        self.epsilon = config.get('epsilon', 0.05)

        self.params.update({'epsilon': self.epsilon})

        self.register_iter_hooks(self.evaluate)
        self.register_exit_hooks(self.evaluate)

    def evaluate(self, loss, done, queries, **kwargs):
        stats = {}
        stats['Adv Loss'] = loss.mean().item()
        if done.sum().item() > 0:
            stats['ASR'] = done.float().mean().item()
            stats['Avg Queries'] = queries[done].mean().item()
            stats['Med Queries'] = queries[done].median().item()
        return stats


class SignHunter(ScoreBasedAttack):
    def __init__(self, config):
        super(SignHunter, self).__init__(config)

        self.DS_R = config.get('down_sampling', 1)

        self.params.update({'down_sampling': self.DS_R})

        if self.DEBUG:
            self.register_iter_hooks(self._cal_cbit_prop)

    def _init_delta(self, x):
        return torch.ones_like(x)

    def _partition(self, segs):
        segs_new = []
        for seg in segs:
            beg, end = seg[: 2]
            if beg < end:
                mid = beg + (end - beg) // 2
                segs_new.append([beg, mid] + seg[2:])
                segs_new.append([mid, end] + seg[2:])
        return segs_new

    def _cal_lazy_grad(self, x, y):
        if x.grad is None:
            x.requires_grad = True
            with torch.enable_grad():
                logits = self.vmodel(x)
                loss = self.criterion(logits, y)
                loss.sum().backward()
            x.requires_grad = False
        return x.grad

    def _cal_cbit_prop(self, x, y, sign, **kwargs):
        grad = self._cal_lazy_grad(x, y)
        omega = grad.abs()
        cbits = grad * sign > 0
        cprop = omega[cbits].sum() / omega.sum()
        return {'Corr bits Prop': cprop.item()}

    def _craft_adv_examples(self, x, s):
        return torch.clamp(x + self.epsilon * s, 0.0, 1.0)

    def _attack_main(self, images, labels):
        device = images.device

        B, C, H, W = images.shape
        DS_H = math.ceil(H / self.DS_R)
        DS_W = math.ceil(W / self.DS_R)

        _down_sample = lambda x: F.interpolate(x, (DS_H, DS_W))
        _up_sample = lambda x: F.interpolate(x, (H, W))
        _flatten = lambda x: x.flatten(1)
        _unflatten = lambda x: x.view(-1, C, DS_H, DS_H)

        _fold = lambda x: _flatten(_down_sample(x))
        _unfold = lambda x: _up_sample(_unflatten(x))

        sign = _fold(self._init_delta(images))
        perturbed = self._craft_adv_examples(images, _unfold(sign))

        logits = self.vmodel(perturbed)
        loss = self.criterion(logits, labels)
        done = self.find_done(logits, labels)

        queries = torch.ones(B).to(device)

        dim = sign.shape[-1]
        sidx, tree_h, segs = 0, 0, [[0, dim]]

        for iter_i in range(self.budget):
            # tree has been traversed
            if not len(segs): break
            # all samples have been attacked
            if done.sum().item() == B: break

            need_query = ~done
            # img idx queried in this iteration
            qidx = torch.where(need_query)[0]

            sign_new = sign[qidx].clone()

            beg = segs[sidx][0]
            end = segs[sidx][1]
            sign_new[:, beg: end] *= -1

            perturbed = self._craft_adv_examples(images[qidx], _unfold(sign_new))

            logits = self.vmodel(perturbed)
            loss_new = self.criterion(logits, labels[qidx])
            done_new = self.find_done(logits, labels[qidx])

            improved = (loss_new > loss[qidx])
            loss[qidx] = improved * loss_new + ~improved * loss[qidx]

            improved = improved.unsqueeze(1)
            sign[qidx] = improved * sign_new + ~improved * sign[qidx]

            done[qidx] = done_new

            queries[qidx] += 1

            self._call_iter_hooks(iter_i, {
                'x': images,
                'y': labels,
                'loss': loss,
                'done': done,
                'queries': queries,
                'sign': _unfold(sign)
            })

            sidx += 1
            if sidx == len(segs):
                sidx = 0; tree_h += 1
                segs = self._partition(segs)

        adv_x = self._craft_adv_examples(images, _unfold(sign))
        adv_y = self.vmodel(adv_x).argmax(dim=-1)

        labels = self.vmodel(images).argmax(dim=-1)

        return self._call_exit_hooks({
            'x': images,
            'y': labels,
            'adv_x': adv_x,
            'adv_y': adv_y,
            'loss': loss,
            'done': done,
            'queries': queries
        })

    def __str__(self):
        return "Sign Hunter"


class AcceleratedSignHunter(SignHunter):
    def __init__(self, config):
        super(AcceleratedSignHunter, self).__init__(config)

    def _attack_main(self, images, labels):
        device = images.device

        B, C, H, W = images.shape
        DS_H = math.ceil(H / self.DS_R)
        DS_W = math.ceil(W / self.DS_R)

        _down_sample = lambda x: F.interpolate(x, (DS_H, DS_W))
        _up_sample = lambda x: F.interpolate(x, (H, W))
        _flatten = lambda x: x.flatten(1)
        _unflatten = lambda x: x.view(-1, C, DS_H, DS_H)

        _fold = lambda x: _flatten(_down_sample(x))
        _unfold = lambda x: _up_sample(_unflatten(x))

        sign = _fold(self._init_delta(images))
        perturbed = self._craft_adv_examples(images, _unfold(sign))

        logits = self.vmodel(perturbed)
        loss = self.criterion(logits, labels)
        done = self.find_done(logits, labels)

        queries = torch.ones(B).to(device)
        finished = queries >= self.budget

        dim = sign.shape[-1]
        sidx, tree_h = 0, 0

        segs = [[0, dim, 0.0]]
        segs_all = [segs for _ in range(B)]

        iter_i = 0
        while len(segs_all[0]):
            # all samples have been reached the budget
            if finished.sum().item() == B: break
            # samples are either attacked or reached the budget
            if (done | finished).sum().item() == B: break

            need_query = ~finished & ~done
            if sidx % 2 == 1:
                for i, segs in enumerate(segs_all):
                    segs[sidx][-1] -= segs[sidx - 1][-1]
                    need_query[i] &= segs[sidx][-1] < 0

            # img idx queried in this iteration
            qidx = torch.where(need_query)[0]

            if len(qidx):
                sign_new = sign[qidx].clone()

                for i, q in enumerate(qidx):
                    beg = segs_all[q][sidx][0]
                    end = segs_all[q][sidx][1]
                    sign_new[i, beg: end] *= -1

                perturbed = self._craft_adv_examples(images[qidx], _unfold(sign_new))

                logits = self.vmodel(perturbed)
                loss_new = self.criterion(logits, labels[qidx])
                done_new = self.find_done(logits, labels[qidx])

                for i, q in enumerate(qidx):
                    segs_all[q][sidx][-1] = (loss[q] - loss_new[i]).item()

                improved = (loss_new > loss[qidx])
                loss[qidx] = improved * loss_new + ~improved * loss[qidx]

                improved = improved.unsqueeze(1)
                sign[qidx] = improved * sign_new + ~improved * sign[qidx]

                done[qidx] = done_new
                finished[qidx] = queries[qidx] >= self.budget

                queries[qidx] += 1

            self._call_iter_hooks(iter_i, {
                'x': images,
                'y': labels,
                'loss': loss,
                'done': done,
                'queries': queries,
                'sign': _unfold(sign)
            })

            sidx += 1
            if sidx == len(segs_all[0]):
                sidx = 0; tree_h += 1
                segs_all_new = []
                for segs in segs_all:
                    segs = [s[: -1] + [abs(s[-1])] for s in segs]
                    segs = sorted(segs, key=lambda s: s[-1])
                    segs = self._partition(segs)
                    segs_all_new.append(segs)
                segs_all = segs_all_new

            iter_i += 1

        adv_x = self._craft_adv_examples(images, _unfold(sign))
        adv_y = self.vmodel(adv_x).argmax(dim=-1)

        labels = self.vmodel(images).argmax(dim=-1)

        return self._call_exit_hooks({
            'x': images,
            'y': labels,
            'adv_x': adv_x,
            'adv_y': adv_y,
            'loss': loss,
            'done': done,
            'queries': queries
        })

    def __str__(self):
        return "Accelerated Sign Hunter"


def build_attacker(config):
    if config['attack'] == 'sh':
        return SignHunter(config)
    elif config['attack'] == 'ash':
        return AcceleratedSignHunter(config)
    else:
        raise NotImplementedError
