import os

import torch
<<<<<<< HEAD
from torch import nn

import torch.nn.functional as F

from torch.autograd import Variable
=======
>>>>>>> 76ab7df4b4f8f2c4eb5b644154c84548fe4b40a3

import numpy as np

VERBOSE = True

<<<<<<< HEAD
USE_CUDA = True

=======
>>>>>>> 76ab7df4b4f8f2c4eb5b644154c84548fe4b40a3

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

<<<<<<< HEAD
    def __init__(self, opt, initial_lr, num_iterations=1000):
=======
    def __init__(self, opt, initial_lr, num_iterations=2000):
>>>>>>> 76ab7df4b4f8f2c4eb5b644154c84548fe4b40a3
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
<<<<<<< HEAD


class CosineSimilarityRegressionLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityRegressionLoss, self).__init__()

    def forward(self, vec1, vec2, y):
        mse = nn.MSELoss()
        y_hat = F.cosine_similarity(vec1, vec2)
        return mse(y_hat, y)


class CosineSimilarityLossWithL2Regularization(nn.Module):
    def __init__(self, cos_sim_margin=0.1, l2_margin=0.1, alpha=0.1):
        super(CosineSimilarityLossWithL2Regularization, self).__init__()
        self.cos_sim_margin = cos_sim_margin
        self.l2_margin = l2_margin
        self.alpha = alpha

    def forward(self, vec1, vec2, y):
        assert vec1.size(0) == vec2.size(0)
        ones = Variable(torch.ones(vec1.size(0), 1))
        if USE_CUDA:
            ones = ones.cuda()
        # l2_1 = torch.clamp(torch.abs(ones - vec1.norm(p=2, dim=1)), max=1.0)
        # l2_2 = torch.clamp(torch.abs(ones - vec2.norm(p=2, dim=1)), max=1.0)
        # l2_1 = l2_1.mean()
        # l2_2 = l2_2.mean()
        l2_1 = F.l1_loss(ones, vec1.norm(p=2, dim=1))
        l2_2 = F.l1_loss(ones, vec2.norm(p=2, dim=1))
        loss = F.cosine_embedding_loss(vec1, vec2, y)
        return loss + self.alpha * (l2_1 + l2_2)
=======
>>>>>>> 76ab7df4b4f8f2c4eb5b644154c84548fe4b40a3
