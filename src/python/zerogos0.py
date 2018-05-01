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

USE_CUDA = False

def set_cuda(val):
    global USE_CUDA
    USE_CUDA = val

import math

import argparse

LR = 0.001

BATCH_SIZE = 16

LONG_EXPOSURE = True


def batch_generator(gen, normalize=False):
    xByShapes, yByShapes = dict(), dict()
    for _, x, y in gen:
        # x, y = np.array(x).reshape(1, len(x), 40), zeroone2oneminusone(y)
        if normalize: x = np.divide(np.add(x, -np.mean(x)), np.std(x))
        if x.shape in xByShapes:
            xByShapes[x.shape].append(x)
            yByShapes[x.shape].append(y)
            X, Y = xByShapes[x.shape], yByShapes[x.shape]
            if len(X) == BATCH_SIZE:
                yield np.array(X), np.array(Y)
                del xByShapes[x.shape]
                del yByShapes[x.shape]
        else:
            xByShapes[x.shape] = [x]
            yByShapes[x.shape] = [y]

    for X, Y in zip(xByShapes.values(), yByShapes.values()):
        yield np.array(X), np.array(Y)


def batch_generator(stream, batch_size=BATCH_SIZE):

    def go2indx(go):
        return [classes.index(go)]

    def pad_seq(seq, max_length=MAX_LENGTH):
        delta = max_length - len(seq)
        left = [PAD for _ in range(delta // 2)]
        right = [PAD for _ in range(delta - delta // 2)]
        seq = left + [AA.aa2index[aa] for aa in seq] + right
        return np.asarray(seq)

    def prepare_batch(sequences, terms, labels):
        b = max(max(map(len, sequences)), 100)
        if USE_CUDA:
            t_go = Variable(torch.LongTensor(np.asarray([[go] for go in terms]))).cuda()
            t_seq = Variable(torch.LongTensor(np.asarray([pad_seq(seq, b) for seq in sequences]))).cuda()
            t_lbl = Variable(torch.LongTensor(np.asarray([[lbl] for lbl in labels]))).cuda()
        else:
            t_go = Variable(torch.LongTensor(np.asarray([[go] for go in terms])))
            t_seq = Variable(torch.LongTensor(np.asarray([pad_seq(seq, b) for seq in sequences])))
            t_lbl = Variable(torch.LongTensor(np.asarray([[lbl] for lbl in labels])))
        return t_seq, t_go, t_lbl

    data = {}
    for seqid, seq, go, lbl in stream:
        key = len(seq)
        if key in data:
            ids, seqs, goz, lbls = zip(*data[key])
            if len(ids) == batch_size:
                yield ids, prepare_batch(seqs, goz, lbls)
                del data[key]
            else:
                data[key].append([seqid, seq, go, lbl])
        else:
            data[key] = [[seqid, seq, go, lbl]]

    for packet in data.values():
        ids, seqs, goz, lbls = zip(*packet)
        yield ids, prepare_batch(seqs, goz, lbls)


# def train(model, gen_xy, length_xy, epoch, num_epochs,
#           history=LossHistory()):
#     pbar = tqdm(total=length_xy)
#
#     for _, (X, Y) in gen_xy:
#         model.fit(x=X, y=Y,
#                   batch_size=BATCH_SIZE,
#                   epochs=max(num_epochs, epoch + 1) if LONG_EXPOSURE else epoch + 1,
#                   verbose=0,
#                   validation_data=None,
#                   initial_epoch=epoch,
#                   callbacks=[history, lrate])
#         pbar.set_description("Training Loss:%.5f" % np.mean(history.losses))
#         pbar.update(len(Y))
#
#     pbar.close()


def zeroone2oneminusone(vec):
    return np.add(np.multiply(np.array(vec), 2), -1)


def oneminusone2zeroone(vec):
    return np.divide(np.add(np.array(vec), 1), 2)


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
    def __init__(self, input_size=23):
        super(DeepSeq, self).__init__()

        self.features = nn.Sequential(

            nn.Conv1d(input_size, 250, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(250, 100, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(250, 100, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(100, 250, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        out = self.features(x)
        kern = out.size()[1:]
        out = F.max_pool1d(out, kernel_size=kern)
        return out


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

    # model = ZeroGO()
    #
    # opt = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), epsilon=1e-8)
    #
    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '%s'" % args.resume)
    #         checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    #         epoch = checkpoint['epoch']
    #         encoder.load_state_dict(checkpoint['encoder'])
    #         decoder.load_state_dict(checkpoint['decoder'])
    #         encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    #         decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
    #     else:
    #         print("=> no checkpoint found at '%s'" % args.resume)

    print("Indexing Data...")
    trn_stream, tst_stream = get_balanced_training_and_validation_streams(db, t0, t1, asp, onto)
    print("Loading Data...")
    tst_data = tst_stream._seq2go

    for i, batch in enumerate(batch_generator(trn_stream)):
        sys.stdout.write("\rLoading {0:.0f}".format(i))

    # for epoch in range(args.init_epoch, num_epochs):
    #
    #     train(model, batch_generator(trn_data, onto, classes), len(trn_data), epoch, num_epochs)
    #
    #     if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
    #         continue
    #
    #     _, y_true, y_pred = predict(model, batch_generator(tst_data, onto, classes), len(tst_data), classes)
    #     loss, prs, rcs, f1s = evaluate(y_true, y_pred, classes)
    #     i = np.argmax(f1s)
    #     f_max = f1s[i]
    #
    #     print("[Epoch %d/%d] (Validation Loss: %.5f, F_max: %.3f, precision: %.3f, recall: %.3f)"
    #           % (epoch + 1, num_epochs, loss, f1s[i], prs[i], rcs[i]))
    #
    #     if f_max < 0.4: continue
    #
    #     model_str = '%s-%d-%.5f-%.2f' % (args.arch, epoch + 1, loss, f_max)
    #     model.save_weights("checkpoints/%s.hdf5" % model_str)
    #     with open("checkpoints/%s.json" % model_str, "w+") as f:
    #         f.write(model.to_json())
