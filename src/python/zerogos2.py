import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from src.python.baselines import *

from src.python.preprocess2 import *

from pymongo import MongoClient

from tqdm import tqdm

import numpy as np

from tempfile import gettempdir

import argparse

LR = 0.001

BATCH_SIZE = 32

LONG_EXPOSURE = True

USE_CUDA = True


def set_cuda(val):
    global USE_CUDA
    USE_CUDA = val


def evaluate(model, gen_xy, length_xy, classes, criterion=nn.BCELoss()):
    net.eval()
    pbar = tqdm(total=length_xy)
    err = 0
    m, n = length_xy, len(classes)
    y_pred = np.zeros((m, n))
    y_true = np.zeros((m, n))
    seq_ids = list()
    for i, (ids, (seq, lbl)) in enumerate(gen_xy):
        seq_ids.extend(ids)
        out = net(seq)
        loss = criterion(out.float(), lbl.float())
        err += loss.data[0]
        y_pred[i:i + len(lbl), ] = out.data.cpu().numpy()
        y_true[i:i + len(lbl), ] = lbl.data.cpu().numpy()
        pbar.set_description("Validation Loss:%.5f" % (err/(i+1)))
        pbar.update(len(lbl))
    pbar.close()
    model.eval()
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
    y_true = y_true[~np.all(y_true == 0, axis=1)]
    prs, rcs, f1s = performance(y_pred, y_true, classes)
    return err/i, prs, rcs, f1s


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

            nn.Conv1d(500, 200, kernel_size=15),
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
        b = seq.size()[0]
        emb = emb.view((b, 23, -1))
        out = self.features(emb)
        out = F.max_pool1d(out, kernel_size=out.size()[2])
        out = out.view((b, -1))
        out = self.classifier(out)
        return out

    # # Loop over
    # def step(self, inputs):
    #     data, label = inputs  # ignore label
    #     outputs = self.model(data)
    #     _, preds = torch.max(outputs.data, 1)
    #     # preds, outputs  are cuda tensors. Right?
    #     return preds, outputs
    #
    # def predict(self, dataloader):
    #     prediction_list = []
    #     for i, batch in enumerate(dataloader):
    #         pred, output = self.step(batch)
    #         prediction_list.append(pred.cpu())


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

    batch = []
    for packet in data.values():
        if len(batch) >= BATCH_SIZE:
            ids, seqs, lbls = zip(*batch)
            yield ids, prepare_batch(seqs, lbls)
            batch = []
        batch.extend(packet)


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
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="How many epochs to train the model?")
    parser.add_argument('--long_exposure', action='store_true', default=False,
                        help="Train in LONG_EXPOSURE mode?")
    parser.add_argument("--arch", type=str, choices=['deepseq', 'inception', 'motifnet'],
                        default="inception", help="Specify the model arch.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")


def train(net, opt, gen_xy, length_xy, exposure=1):

    net.train()

    pbar = tqdm(total=length_xy)

    criterion = nn.BCELoss()

    err = 0

    for i, (_, (seq, lbl)) in enumerate(gen_xy):

        for j in range(exposure):
            opt.zero_grad()
            out = net(seq)
            loss = criterion(out.float(), lbl.float())
            err += loss.data[0]
            loss.backward()
            opt.step()

        pbar.set_description("Training Loss:%.5f" % (err/(i * j + 1)))
        pbar.update(len(lbl))

    pbar.close()


def model_summary(model):
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)


def save_checkpoint(state, loss, fmax):
    filename_late = os.path.join(ckptpath, "%.5f-%.5f-deepseq.tar" % (loss, fmax))
    torch.save(state, filename_late)


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

    ckptpath = args.out_dir

    if USE_CUDA: net = net.cuda()

    opt = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

    model_summary(net)

    print("#####################################################################")

    print("Indexing Data...")
    trn_stream, tst_stream = get_random_training_and_validation_streams(db, asp)
    print("Loading Data...")
    trn_data = load_data(trn_stream)
    tst_data = load_data(tst_stream)

    if args.long_exposure:
        exp = 20
        num_epochs = 20
    else:
        exp = 1
        num_epochs = 80

    for epoch in range(args.init_epoch, num_epochs):

        train(net, opt, batch_generator(trn_data, onto, classes), len(trn_data), exposure=exp)

        # if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
        #     continue

        loss, prs, rcs, f1s = evaluate(net, batch_generator(tst_data, onto, classes), len(tst_data), classes)
        i = np.argmax(f1s)
        f_max = f1s[i]

        print("[Epoch %d/%d] (Validation Loss: %.5f, F_max: %.3f, precision: %.3f, recall: %.3f)"
              % (epoch + 1, num_epochs, loss, f1s[i], prs[i], rcs[i]))

        if f_max < 0.4: continue

        save_checkpoint({
            'epoch': epoch,
            'net': net.state_dict(),
            'opt': opt.state_dict()
        }, loss, f_max)
