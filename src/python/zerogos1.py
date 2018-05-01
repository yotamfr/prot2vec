import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from src.python.baselines import *

from src.python.preprocess2 import *

from pymongo import MongoClient

from tqdm import tqdm

import tensorflow as tf

### Keras
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

USE_CUDA = True


def set_cuda(val):
    global USE_CUDA
    USE_CUDA = val

import math

import argparse

LR = 0.001

BATCH_SIZE = 32

LONG_EXPOSURE = True


def calc_loss(y_true, y_pred):
    return np.mean([log_loss(y, y_hat) for y, y_hat in zip(y_true, y_pred) if np.any(y)])


def predict(model, gen_xy, length_xy, classes):
    pbar = tqdm(total=length_xy, desc="Predicting...")
    i, m, n = 0, length_xy, len(classes)
    ids = list()
    y_pred, y_true = np.zeros((m, n)), np.zeros((m, n))
    for i, (keys, (X, Y)) in enumerate(gen_xy):
        k = len(Y)
        ids.extend(keys)
        y_hat, y = model.predict(X), Y
        y_pred[i:i + k, ] = y_hat
        y_true[i:i + k, ] = y
        pbar.update(k)
    pbar.close()
    return ids, y_true, y_pred


def evaluate(y_true, y_pred, classes):
    y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
    y_true = y_true[~np.all(y_true == 0, axis=1)]
    prs, rcs, f1s = performance(y_pred, y_true, classes)
    return calc_loss(y_true, y_pred), prs, rcs, f1s


class ZeroGO(nn.Module):
    def __init__(self, seq_model, go_model, score_model):
        super(ZeroGO, self).__init__()
        self.seq_feats = seq_model
        self.go_feats = go_model
        self.score = score_model

    def forward(self, inp_seq, inp_go):
        prot = self.seq_feats(inp_seq)
        govec = self.go_feats(inp_go)
        logits = self.score(prot, govec)
        return logits


class DeepSeq(nn.Module):
    def __init__(self, classes):
        super(DeepSeq, self).__init__()

        self.embedding = nn.Embedding(26, 23)

        self.features = nn.Sequential(

            nn.Conv1d(23, 500, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(200, 200, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(200, 200, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(200, 200, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(200, len(classes)),
            nn.BatchNorm1d(len(classes)),
            nn.Sigmoid()
        )

    def forward(self, seq):
        emb = self.embedding(seq)
        emb = emb.view((BATCH_SIZE, 23, -1))
        out = self.features(emb)
        out = F.max_pool1d(out, kernel_size=out.size()[2])
        out = out.view((BATCH_SIZE, -1))
        out = self.classifier(out)
        return out


def batch_generator(stream, onto, classes):

    s_cls = set(classes)
    data = dict()

    def labels2vec(lbl):
        y = np.zeros(len(classes))
        for go in onto.propagate(lbl, include_root=False):
            if go not in s_cls:
                continue
            y[classes.index(go)] = 1
        return y

    def pad_seq(seq, max_length=MAX_LENGTH):
        delta = max_length - len(seq)
        left = [PAD for _ in range(delta // 2)]
        right = [PAD for _ in range(delta - delta // 2)]
        seq = left + [AA.aa2index[aa] for aa in seq] + right
        return np.asarray(seq)

    def prepare_batch(sequences, labels):
        b = max(map(len, sequences)) + 100
        Y = np.asarray([labels2vec(lbl) for lbl in labels])
        X = np.asarray([pad_seq(seq, b) for seq in sequences])
        inp_var = Variable(torch.LongTensor(X))
        lbl_var = Variable(torch.LongTensor(Y))
        if USE_CUDA:
            inp_var = inp_var.cuda()
            lbl_var = lbl_var.cuda()
        return inp_var, lbl_var

    for k, x, y in stream:
        lx = len(x)
        if lx in data:
            data[lx].append([k, x, y])
            ids, seqs, lbls = zip(*data[lx])
            if len(seqs) == BATCH_SIZE:
                yield ids, prepare_batch(seqs, lbls)
                del data[lx]
        else:
            data[lx] = [[k, x, y]]

    for packet in data.values():
        ids, seqs, lbls = zip(*packet)
        yield ids, prepare_batch(seqs, lbls)


class GoVec(nn.Module):
    def __init__(self, vocabulary_size, emb_weights, requires_grad=False):
        super(GoVec, self).__init__()
        embedding_size = emb_weights.shape[1]
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(emb_weights).float())
        self.embedding.requires_grad = requires_grad

    def forward(self, go):
        return self.embedding(go)


class GoScore(nn.Module):
    def __init__(self, method, size_seq, size_go):
        super(GoScore, self).__init__()

        self.method = method

        if self.method == 'general':
            self.attn = nn.Linear(size_go, size_seq)

        elif self.method == 'concat':
            self.attn = nn.Linear(size_seq + size_go, size_seq)
            self.v = nn.Parameter(torch.FloatTensor(1, size_seq))

    def forward(self, deepseq, govec):

        if self.method == 'dot':
            energy = torch.dot(deepseq.view(-1), deepseq.view(-1))
            return energy

        elif self.method == 'general':
            energy = self.attn(govec)
            energy = torch.dot(deepseq.view(-1), energy.view(-1))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((deepseq, govec), 1))
            energy = self.v.dot(energy)
            return energy


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB"),
    parser.add_argument("--aspect", type=str, choices=['F', 'P', 'C'],
                        default="F", help="Specify the ontology aspect.")
    parser.add_argument("--init_epoch", type=int, default=0,
                        help="Which epoch to start training the model?")
    parser.add_argument("--num_epoch", type=int, default=200,
                        help="Which epoch to end training the model?")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-e", "--eval_every", type=int, default=10,
                        help="How often to evaluate on the validation set.")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="How many epochs to train the model?")
    parser.add_argument('--long_exposure', action='store_true', default=False,
                        help="Train in LONG_EXPOSURE mode?")
    parser.add_argument("--arch", type=str, choices=['deepseq', 'inception', 'motifnet'],
                        default="inception", help="Specify the model arch.")


def train(net, opt, gen_xy, length_xy):

    net.train()

    pbar = tqdm(total=length_xy)

    criterion = nn.BCELoss()

    err = 0

    for i, (_, (seq, lbl)) in enumerate(gen_xy):

        for j in range(20):
            opt.zero_grad()
            out = net(seq)
            loss = criterion(out.float(), lbl.float())
            err += loss.data[0]
            loss.backward()

        pbar.set_description("Training Loss:%.5f" % (err/(i*j+1)))
        pbar.update(len(lbl))

    pbar.close()


def model_summary(model):
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    asp = args.aspect  # default: Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(asp)

    classes = onto.classes
    classes.remove(onto.root)
    assert onto.root not in classes

    net = DeepSeq(classes)

    if USE_CUDA: net = net.cuda()

    opt = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

    model_summary(net)

    print("#####################################################################")

    print("Indexing Data...")
    trn_stream, tst_stream = get_balanced_training_and_validation_streams(db, t0, t1, asp, onto, classes)
    print("Loading Data...")
    trn_data = load_data(trn_stream)
    tst_data = load_data(tst_stream)

    for epoch in range(args.init_epoch, args.num_epochs):

        train(net, opt, batch_generator(trn_data, onto, classes), len(trn_data))

        if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
            continue

        # _, y_true, y_pred = predict(model, batch_generator(tst_data, onto, classes), len(tst_data), classes)
        # loss, prs, rcs, f1s = evaluate(y_true, y_pred, classes)
        # i = np.argmax(f1s)
        # f_max = f1s[i]
        #
        # print("[Epoch %d/%d] (Validation Loss: %.5f, F_max: %.3f, precision: %.3f, recall: %.3f)"
        #       % (epoch + 1, num_epochs, loss, f1s[i], prs[i], rcs[i]))
        #
        # if f_max < 0.4: continue
        #
        # model_str = '%s-%d-%.5f-%.2f' % (args.arch, epoch + 1, loss, f_max)
        # model.save_weights("checkpoints/%s.hdf5" % model_str)
        # with open("checkpoints/%s.json" % model_str, "w+") as f:
        #     f.write(model.to_json())
