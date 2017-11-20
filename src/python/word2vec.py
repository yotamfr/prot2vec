import os
import sys
import operator
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

from itertools import combinations

from gensim import matutils

from pymongo.errors import CursorNotFound
from pymongo import MongoClient
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import matplotlib.pyplot as plt
except ImportError as err:
    plt = None
    print(err)

from tempfile import gettempdir

import argparse

np.random.seed(1809)


class BatchLoader(object):
    def __init__(self, win_size, batch_size, train):
        self.win_size = win_size
        self.batch_size = batch_size
        self.batch_buffer = np.ndarray(shape=(batch_size,), dtype=np.int32)
        self.labels_buffer = np.ndarray(shape=(batch_size,), dtype=np.int32)
        self.train = train

    def __iter__(self):

        if self.train:
            self.stream = map(lambda p: p['sequence'], collection.find({}))
        else:
            self.stream = map(lambda p: p['sequence'], collection.aggregate([{"$sample": {"size": args.size_test}}]))

        seq = self._get_sequence()
        seq_pos = 0

        batch_buffer, labels_buffer = self.batch_buffer, self.labels_buffer

        while len(seq):

            batch_buffer, labels_buffer, seq_pos, batch_pos = \
                self._get_batch(seq, batch_buffer, labels_buffer, batch_pos=0, seq_pos=seq_pos)

            if seq_pos == 0:  # seq finished
                try:
                    seq = self._get_sequence()
                except CursorNotFound:
                    break

                batch_buffer, labels_buffer, seq_pos, batch_pos = \
                    self._get_batch(seq, batch_buffer, labels_buffer, batch_pos=batch_pos, seq_pos=0)

            else:
                yield batch_buffer, labels_buffer

    def _get_batch(self, seq, batch, labels, batch_pos=0, seq_pos=0):
        return batch, labels, seq_pos, batch_pos

    def _get_sequence(self):
        return ""


class WindowBatchLoader(BatchLoader):
    def __init__(self, win_size, batch_size, train=True):
        super(WindowBatchLoader, self).__init__(win_size, batch_size, train)
        self.mask = np.array(range(2 * win_size + 1)) != win_size
        self.batch_buffer = np.ndarray(shape=(batch_size, win_size * 2), dtype=np.int32)
        self.labels_buffer = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    def _get_sequence(self):
        return np.array(list(map(lambda aa: dictionary[aa], next(self.stream))))

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

    def _get_sequence(self):
        return next(self.stream, None)

    def _get_batch(self, seq, batch, labels, batch_pos=0, seq_pos=0):
        batch_size = self.batch_size
        possible_context_sizes = range(1, self.win_size + 1)
        while batch_pos < batch_size:
            win = np.random.choice(possible_context_sizes)
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
                labels[batch_pos] = dictionary[seq[label_pos]]
                batch[batch_pos] = dictionary[seq[seq_pos]]
                batch_pos += 1
            seq_pos += 1

        return batch, labels, seq_pos, batch_pos


def get_negative_samples(words):
    return np.array([np.random.choice(aa_unlike[w[0]], 1) for w in words])


def train(model, train_loader, test_loader):
    # Hyper Parameters

    num_epochs = 2
    learning_rate = 0.003

    # Loss and Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()
    train_loss = 0

    for epoch in range(num_epochs):

        # pbar = tqdm(range(len(train_dataset)), "training ... ")

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
                print('Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f'
                      % (epoch + 1, num_epochs, train_loss / args.steps_per_stats, test_loss / i))
                train_loss = 0

            # pbar.update(batch_size)

        # pbar.close()

        model.save()            # Save the Trained Model


def get_loss(word, context, model, criterion):
    word_tag = get_negative_samples(word)
    c = Variable(torch.from_numpy(context).long())
    w = Variable(torch.from_numpy(word).long())
    w_tag = Variable(torch.from_numpy(word_tag).long())

    l_pos = Variable(torch.from_numpy(np.ones((word.shape[0], 1))).float())
    l_neg = Variable(torch.from_numpy(np.zeros((word.shape[0], 1))).float())

    p_pos = model((w, c))
    p_neg = 1 - model((w_tag, c))
    return criterion(p_pos, l_pos) + criterion(p_neg, l_neg)


def embeddings(w2v):
    return w2v.emb_w.weight.data.numpy()


class Word2VecAPI(object):
    def __init__(self, w2v):
        self.w2v = w2v

    @property
    def embeddings(self):
        return embeddings(self.w2v)

    def __getitem__(self, aa):
        return self.embeddings[dictionary[aa]]

    def __contains__(self, aa):
        return aa in dictionary

    @property
    def vocab(self):
        keys = list(dictionary.keys())
        values = [self[aa] for aa in keys]
        return dict(zip(keys, values))

    def similarity(self, aa1, aa2):
        return np.dot(matutils.unitvec(self[aa1]), matutils.unitvec(self[aa2]))
        # return self.sim[dictionary[aa1], dictionary[aa2]]


class Word2VecImpl(nn.Module):

    def __init__(self, emb_size):
        super(Word2VecImpl, self).__init__()
        self.emb_w = nn.Embedding(vocabulary_size, emb_size)
        self.emb_c = nn.Embedding(vocabulary_size, emb_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x


class CBOW(Word2VecImpl):

    def __init__(self, emb_size):
        super(CBOW, self).__init__(emb_size)

    def forward(self, x):
        word, context = x
        w_emb = self.emb_w(word)
        c_emb = self.emb_c(context)
        out = torch.bmm(w_emb, c_emb.transpose(1, 2))
        out = out.sum(2)
        out = self.sig(out)
        return out

    def save(self):
        torch.save(self.state_dict(), '%s/cbow.pkl' % ckptpath)


class SkipGram(Word2VecImpl):
    def __init__(self, emb_size):
        super(SkipGram, self).__init__(emb_size)

    def forward(self, x):
        word, context = x
        w_emb = self.emb_w(word)
        c_emb = self.emb_c(context)
        out = torch.bmm(w_emb, c_emb.transpose(1, 2))
        out = self.sig(out).mean(2)
        return out

    def save(self):
        torch.save(self.state_dict(), '%s/sg.pkl' % ckptpath)


def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
    plt.show()


def pca(embeddings):
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


def tsne(embeddings):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    return tsne.fit_transform(embeddings[:vocabulary_size, :])


def plot(low_dim_embs):
    labels = [reverse_dictionary[i] for i in range(vocabulary_size)]
    path = os.path.join(ckptpath, 'plot.png')
    plot_with_labels(low_dim_embs, labels, path)
    print("Saving to %s" % path)


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


def add_arguments(parser):
    parser.add_argument("-w", "--win_size", type=int, required=True,
                        help="Give the length of the context window.")
    parser.add_argument("-d", "--emb_dim", type=int, required=True,
                        help="Give the dimension of the embedding vector.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="Give the size of bach to use when training.")
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("-s", "--source", type=str, choices=['uniprot', 'sprot'],
                        default="sprot", help="Give source name.")
    parser.add_argument("-m", "--model", type=str, choices=['cbow', 'skipgram'],
                        default="cbow", help="Choose what type of model to use.")
    parser.add_argument("-i", "--input_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the input directory.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument('--train', action='store_true', default=False,
                        help="Specify whether to retrain the model.")
    parser.add_argument("-v", '--verbose', action='store_true', default=False,
                        help="Run in verbose mode.")
    parser.add_argument("--steps_per_stats", type=int, default=1000,
                        help="How many training steps to do per stats logging, save.")
    parser.add_argument("--size_test", type=int, default=500,
                        help="The number of sequences sampled to create the test set.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    aa_sim = pd.read_csv('%s/aa_sim.csv' % args.input_dir)
    aa_unlike = [np.where(aa_sim.loc[[w], :] == 0)[1] for w in range(0, 25)]
    AA = aa_sim.columns
    reverse_dictionary = dict(zip(range(len(AA)), AA))
    dictionary = dict(zip(AA, range(len(AA))))
    vocabulary_size = 25
    n_clstr = 8

    client = MongoClient(args.mongo_url)
    collection = client['prot2vec'][args.source]

    ckptpath = args.out_dir
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)

    if args.train:
        if args.model == 'cbow':
            w2v = CBOW(args.emb_dim)
            train_loader = WindowBatchLoader(args.win_size, args.batch_size // 2)
            test_loader = WindowBatchLoader(args.win_size, args.batch_size // 2, False)
            train(w2v, train_loader, test_loader)

        elif args.model == 'skipgram':
            w2v = SkipGram(args.emb_dim)
            train_loader = WindowBatchLoader(args.win_size, args.batch_size // 2)
            test_loader = WindowBatchLoader(args.win_size, args.batch_size // 2, False)
            train(w2v, train_loader, test_loader)

        else:
            print("Unknown model")
            exit(1)

    else:
        if args.model == 'cbow':
            pass

        elif args.model == 'skipgram':
            pass

        else:
            print("Unknown model")
            exit(1)

    if args.verbose and plt:
        plot(tsne(embeddings(w2v)))
        plot(pca(embeddings(w2v)))
    if args.verbose:
        api = Word2VecAPI(w2v)
        print(clstr_stats(api, n_clstr))
