import os
import sys
import operator
import numpy as np
import pandas as pd

from shutil import copyfile

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

from itertools import combinations

from pymongo.errors import CursorNotFound
from pymongo import MongoClient

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.matutils import unitvec

try:
    import matplotlib.pyplot as plt
except ImportError as err:
    plt = None
    print(err)

from tempfile import gettempdir


import argparse

aa_sim = pd.read_csv('Data/aa_sim.csv')
aa_unlike = [np.where(aa_sim.loc[[w], :] == 0)[1] for w in range(0, 25)]
AA = aa_sim.columns
dictionary, reverse_dictionary = dict(zip(AA, range(25))), dict(zip(range(25), AA))
vocabulary_size = len(AA)
n_clstr = 8

import json
print(json.dumps(dictionary, indent=1))

assert vocabulary_size == 25

verbose = False

print("WARNING! Deprecated. Please use aa_predict.py")


class BatchLoader(object):
    def __init__(self, win_size, batch_size, train):
        self.win_size = win_size
        self.batch_size = batch_size
        self.train = train
        self.batch_buffer = np.ndarray([])
        self.labels_buffer = np.ndarray([])
        self.test_set = list(collection_test.aggregate([{"$sample": {"size": size_test}}]))

    def __iter__(self):

        if self.train:
            self.stream = collection_train.aggregate([{"$sample": {"size": size_train}}])
        else:
            self.stream = (seq for seq in self.test_set)

        i, n = 1, size_test
        seq = self._get_sequence()
        seq_pos = 0

        batch_buffer, labels_buffer = self.batch_buffer, self.labels_buffer

        while True:

            batch_buffer, labels_buffer, seq_pos, batch_pos = \
                self._get_batch(seq, batch_buffer, labels_buffer, batch_pos=0, seq_pos=seq_pos)

            if seq_pos == 0:  # seq finished
                try:
                    if verbose:
                        sys.stdout.write("\r{0:.0f}%".format(100.0 * i / n))
                    seq = self._get_sequence()
                    i += 1
                except (CursorNotFound, StopIteration) as e:
                    print(e)
                    break

                batch_buffer, labels_buffer, seq_pos, batch_pos = \
                    self._get_batch(seq, batch_buffer, labels_buffer, batch_pos=batch_pos, seq_pos=0)

            else:
                yield np.random.permutation(batch_buffer), np.random.permutation(labels_buffer)

    def _get_batch(self, seq, batch, labels, batch_pos=0, seq_pos=0):
        return batch, labels, seq_pos, batch_pos

    def _get_sequence(self):
        return ''


class WindowBatchLoader(BatchLoader):
    def __init__(self, win_size, batch_size, train=True):
        super(WindowBatchLoader, self).__init__(win_size, batch_size, train)
        self.mask = np.array(range(2 * win_size + 1)) != win_size
        self.batch_buffer = np.ndarray(shape=(batch_size, win_size * 2), dtype=np.int32)
        self.labels_buffer = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    def _get_sequence(self):
        return np.array(list(map(lambda aa: dictionary[aa], next(self.stream)['sequence'])))

    def _get_batch(self, seq, batch, labels, batch_pos=0, seq_pos=0):
        win = self.win_size
        mask = self.mask
        if seq_pos == 0:
            seq_pos = win
        batch_size = self.batch_size
        while batch_pos < batch_size:
            if seq_pos + win >= len(seq):  # seq finished before batch
                return batch, labels, 0, batch_pos
            if batch_pos == batch_size:  # batch finished before seq
                break
            start = seq_pos - win
            end = seq_pos + win + 1
            context = seq[start:end]
            batch[batch_pos, :] = context[mask]
            labels[batch_pos] = [context[win]]
            batch_pos += 1
            seq_pos += 1

        return batch, labels, seq_pos, batch_pos


class SkipGramBatchLoader(BatchLoader):
    def __init__(self, win_size, batch_size, train=True):
        super(SkipGramBatchLoader, self).__init__(win_size, batch_size, train)
        self.batch_buffer = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        self.labels_buffer = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    def _get_sequence(self):
        return next(self.stream)['sequence']

    def _get_batch(self, seq, batch, labels, batch_pos=0, seq_pos=0):
        batch_size = self.batch_size
        while batch_pos < batch_size:
            win = np.random.choice(self.win_size) + 1
            for offset in range(-win, win + 1):
                label_pos = seq_pos + offset
                if offset == 0 or label_pos < 0:
                    continue
                if label_pos >= len(seq):
                    continue
                if batch_pos == batch_size:  # batch finished before seq
                    break
                if seq_pos == len(seq):  # seq finished before batch
                    return batch, labels, 0, batch_pos
                labels[batch_pos][0] = dictionary[seq[label_pos]]
                batch[batch_pos][0] = dictionary[seq[seq_pos]]
                batch_pos += 1
            seq_pos += 1

        return batch, labels, seq_pos, batch_pos


def get_negative_samples(words):
    return np.array([np.random.choice(aa_unlike[w[0]], 1) for w in words])


def save_checkpoint(state, is_best):
    filename_late = "%s/w2v_%s_latest.tar" % (ckptpath, arch)
    filename_best = "%s/w2v_%s_best.tar" % (ckptpath, arch)
    torch.save(state, filename_late)
    if is_best:
        copyfile(filename_late, filename_best)


def get_loss2(word, context, model, criterion):
    c = Variable(torch.from_numpy(context).long())
    w = Variable(torch.from_numpy(word).long())
    p = model((w, c))
    _ = Variable(torch.from_numpy(np.ones(word.shape[0])))
    return criterion(p, _)


def get_loss1(word, context, model, criterion):
    word_tag = get_negative_samples(word)
    c = Variable(torch.from_numpy(context).long())
    w = Variable(torch.from_numpy(word).long())
    w_tag = Variable(torch.from_numpy(word_tag).long())
    l_pos = Variable(torch.from_numpy(np.ones((word.shape[0], 1))).float())
    l_neg = Variable(torch.from_numpy(np.zeros((word.shape[0], 1))).float())
    p_pos = model((w, c))
    p_neg = 1 - model((w_tag, c))
    return criterion(p_pos, l_pos) + criterion(p_neg, l_neg)


def train_w2v(model, train_loader, test_loader, criterion=nn.MSELoss(), get_loss=get_loss1):
    # Hyper Parameters

    num_epochs = args.num_epochs

    # Loss and Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    train_loss = 0
    best_loss = np.inf
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '%s'" % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '%s' (epoch %s)" %
                  (args.resume, checkpoint['epoch'] + 1))
        else:
            print("=> no checkpoint found at '%s'" % args.resume)

    for epoch in range(start_epoch, num_epochs):

        for step, (context, word) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            loss = get_loss(word, context, model, criterion)
            train_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            # loss = p_pos.log() + p_neg.log()

            if (step + 1) % args.steps_per_stats == 0:
                test_loss = 0
                for i, (c, w) in enumerate(test_loader):
                    loss = get_loss(w, c, model, criterion)
                    test_loss += loss.data[0]
                print('Epoch [%d/%d], Train Loss: %.5f, Test Loss: %.5f'
                      % (epoch + 1, num_epochs, train_loss / args.steps_per_stats, test_loss / i))
                train_loss = 0

                # remember best prec@1 and save checkpoint
                is_best = best_loss > test_loss
                best_loss = min(best_loss, test_loss)
                save_checkpoint({
                    'epoch': epoch,
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, is_best)


def embeddings(w2v):
    return w2v.emb.weight.data.numpy()


class Word2VecAPI(object):
    def __init__(self, w2v):
        self._emb = np.copy(embeddings(w2v))

    @property
    def embeddings(self):
        return np.copy(self._emb)

    def __getitem__(self, aa):
        return self._emb[dictionary[aa]]

    def __contains__(self, aa):
        return aa in dictionary

    @property
    def vocab(self):
        keys = list(dictionary.keys())
        values = [self[aa] for aa in keys]
        return dict(zip(keys, values))

    def similarity(self, aa1, aa2):
        return np.dot(unitvec(self[aa1]), unitvec(self[aa2]))


class Word2VecImpl(nn.Module):

    def __init__(self, emb_size):
        super(Word2VecImpl, self).__init__()
        self.emb_w = nn.Embedding(vocabulary_size, emb_size)
        self.emb_c = nn.Embedding(vocabulary_size, emb_size)

    @property
    def emb(self):
        return self.emb_w

    def forward(self, x):
        return x


class LogLoss(_Loss):

    def forward(self, input, target):
        out = -input.log().mean()
        return out


class SoftMax(Word2VecImpl):
    def __init__(self, emb_size):
        super(SoftMax, self).__init__(emb_size)

    def forward(self, x):
        word, context = x
        batch_size = word.data.shape[0]
        vocab = np.array([list(range(vocabulary_size))
                          for _ in range(batch_size)])
        vocab = Variable(torch.from_numpy(vocab).long())
        v_emb = self.emb_c(vocab)
        w_emb = self.emb_w(word).transpose(1, 2)
        c_emb = self.emb_c(context)
        nom = torch.exp(torch.bmm(c_emb, w_emb))
        dnom = torch.exp(torch.bmm(v_emb, w_emb))
        dnom = dnom.sum(1).pow(-1).unsqueeze(1)
        out = nom.bmm(dnom).view(-1)
        return out


class CBOW(Word2VecImpl):

    def __init__(self, emb_size):
        super(CBOW, self).__init__(emb_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        word, context = x
        w_emb = self.emb_w(word)
        c_emb = self.emb_c(context)
        out = torch.bmm(w_emb, c_emb.transpose(1, 2))
        out = out.sum(2)
        out = self.sig(out)
        return out


class SkipGram(Word2VecImpl):
    def __init__(self, emb_size):
        super(SkipGram, self).__init__(emb_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        word, context = x
        w_emb = self.emb_w(word)
        c_emb = self.emb_c(context)
        out = torch.bmm(w_emb, c_emb.transpose(1, 2))
        out = self.sig(out).mean(2)
        return out


def pca(embeddings):
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


def tsne(embeddings):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    return tsne.fit_transform(embeddings)


def plot(low_dim_embs, fname=None):
    labels = [reverse_dictionary[i] for i in range(vocabulary_size)]
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right', va='bottom')
    if fname:
        fpath = os.path.join(ckptpath, fname)
        print("Saving to %s" % fpath)
        plt.savefig(fpath)
    plt.show()


def kmeans(w2v, k):
    keys = np.array(list(w2v.vocab.keys()))
    vectors = np.array([w2v[aa] for aa in keys])
    km = KMeans(n_clusters=k).fit(vectors)
    return keys, km.labels_


def clstr_stats(w2v, k):
    keys, labels = kmeans(w2v, k)
    clstr = '\n'.join("cluster %s: %s" %
                      (lbl, ' '.join(keys[labels == lbl]))
                      for lbl in np.unique(labels))
    cs = combinations(keys, 2)
    ds = {c: w2v.similarity(c[0], c[1]) for c in cs}
    hi_i = max(ds.items(), key=operator.itemgetter(1))[0]
    lo_i = min(ds.items(), key=operator.itemgetter(1))[0]
    av = np.mean(list(ds.values()))
    hi_s = "highest similarity: sim(%s, %s)=%s" % (hi_i[0], hi_i[1], ds[hi_i])
    lo_s = "lowest similarity: sim(%s, %s)=%s" % (lo_i[0], lo_i[1], ds[lo_i])
    av_s = "average similarity: %s" % av
    return '\n'.join([clstr, hi_s, lo_s, av_s])


def nn_stats(w2v):
    for i in range(vocabulary_size):
        aa = reverse_dictionary[i]
        top_k = 3  # number of nearest neighbors
        nearest = sorted(list(range(vocabulary_size)),
                         key=lambda o: -w2v.similarity(aa, reverse_dictionary[o]))
        log_str = 'Nearest to %s:' % aa
        for k in range(1, top_k+1):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)


def add_arguments(parser):
    parser.add_argument("-w", "--win_size", type=int, required=True,
                        help="Give the length of the context window.")
    parser.add_argument("-d", "--emb_dim", type=int, required=True,
                        help="Give the dimension of the embedding vector.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Give the size of bach to use when training.")
    parser.add_argument("-e", "--num_epochs", type=int, default=5,
                        help="Give the number of epochs to use when training.")
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("-a", "--arch", type=str, choices=['cbow', 'sg', 'sf'],
                        default="cbow", help="Choose what type of model to use.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument("-v", '--verbose', action='store_true', default=False,
                        help="Run in verbose mode.")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--steps_per_stats", type=int, default=1000,
                        help="How many training steps to do per stats logging, save.")
    parser.add_argument("--size_train", type=int, default=50000,
                        help="The number of sequences sampled to create the test set.")
    parser.add_argument("--size_test", type=int, default=1000,
                        help="The number of sequences sampled to create the train set.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    collection_train = db['uniprot']
    collection_test = db['sprot']

    arch = args.arch

    ckptpath = args.out_dir
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)

    size_train = args.size_train
    size_test = args.size_test

    if arch == 'cbow':
        w2v = CBOW(args.emb_dim)
        train_loader = WindowBatchLoader(args.win_size, args.batch_size // 2)
        test_loader = WindowBatchLoader(args.win_size, args.batch_size // 2, False)
        train_w2v(w2v, train_loader, test_loader)

    elif arch == 'sg':
        w2v = SkipGram(args.emb_dim)
        train_loader = SkipGramBatchLoader(args.win_size, args.batch_size // 2)
        test_loader = SkipGramBatchLoader(args.win_size, args.batch_size // 2, False)
        train_w2v(w2v, train_loader, test_loader)

    elif arch == 'sf':
        w2v = SoftMax(args.emb_dim)
        train_loader = SkipGramBatchLoader(args.win_size, args.batch_size // 2)
        test_loader = SkipGramBatchLoader(args.win_size, args.batch_size // 2, False)
        train_w2v(w2v, train_loader, test_loader, LogLoss(), get_loss=get_loss2)

    else:
        print("Unknown model")
        exit(1)

    if args.verbose:
        api = Word2VecAPI(w2v)
        print(clstr_stats(api, n_clstr))
        print(nn_stats(api))
    if args.verbose and plt:
        plot(tsne(api.embeddings), 'w2v_%s_tsne.png' % arch)
        plot(pca(api.embeddings), 'w2v_%s_pca.png' % arch)
