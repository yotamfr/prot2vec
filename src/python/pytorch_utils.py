import os

import torch

import numpy as np

VERBOSE = True


def model_summary(model):
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)


def save_checkpoint(state, loss, prefix, ckptpath):
    filename_late = os.path.join(ckptpath, "%s_%.5f.tar" % (prefix, loss))
    torch.save(state, filename_late)


def adjust_learning_rate(initial, optimizer, epoch, factor=0.1):
    lr = max(initial * (factor ** (epoch // 2)), 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# https://github.com/pytorch/pytorch/issues/2830
def optimizer_cuda(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


class AdaptiveLR(object):

    def __init__(self, opt, initial_lr, num_iterations=2000):
        self._lr = initial_lr
        self.opt = opt
        self.losses = []
        self.window = num_iterations
        self.min_lr = 0.0001
        self.factor = 0.5

    def update(self, loss):
        losses = self.losses
        while len(losses) > self.window:
            losses.pop(0)
        losses.append(loss)
        if len(losses) < self.window:
            return
        avg_old = np.mean(losses[:self.window//2])
        avg_new = np.mean(losses[self.window//2:])
        if avg_new < avg_old:
            return
        self.lr = max(self.lr * self.factor, self.min_lr)
        self.losses = []     # restart loss count

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, val):
        if VERBOSE:
            print("resetting LR: %s -> %s" % (self._lr, val))
        set_learning_rate(val, self.opt)
        self._lr = val


def shuffle(data, labels):
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    return data[s], labels[s]
