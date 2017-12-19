import os
import sys
import numpy as np
import pandas as pd

from shutil import copyfile

import torch
import torch.nn as nn
from torch.autograd import Variable

from pymongo.errors import CursorNotFound
from pymongo import MongoClient

from sklearn.metrics import f1_score
from sklearn import preprocessing

from tempfile import gettempdir

import argparse

aa_sim = pd.read_csv('Data/aa_sim.csv')
aa_unlike = [np.where(aa_sim.loc[[w], :] == 0)[1] for w in range(0, 25)]
AA = aa_sim.columns
dictionary, reverse_dictionary = dict(zip(AA, range(25))), dict(zip(range(25), AA))
vocabulary_size = len(AA)
import json
print(json.dumps(dictionary, indent=1))

aa_feat = pd.read_csv('Data/aa_feat.csv')
min_max_scaler = preprocessing.MinMaxScaler()
aa_feats = aa_feat.loc[:, aa_feat.columns != 'Short'].as_matrix()
aa_feats = min_max_scaler.fit_transform(aa_feats)
num_feats = aa_feats.shape[1]
print(num_feats)


assert vocabulary_size == 25

verbose = False


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
                # yield np.copy(batch_buffer), np.copy(labels_buffer)

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


class Model(nn.Module):

    def __init__(self, emb_size, win_size):
        super(Model, self).__init__()

        self.emb1 = nn.Embedding(vocabulary_size, num_feats)
        self.emb1.weight = nn.Parameter(torch.from_numpy(aa_feats).float())
        self.emb1.requires_grad = False

        self.win_size = win_size
        self.inp_size = emb_size + num_feats
        self.emb2 = nn.Embedding(vocabulary_size, emb_size)

    def emb(self, x):
        emb1 = self.emb1(x)
        emb2 = self.emb2(x)
        emb = torch.cat((emb1, emb2), 2)
        return emb.unsqueeze(1)


class CNN(Model):
    def __init__(self, emb_size, win_size):
        super(CNN, self).__init__(emb_size, win_size)

        hidden_size = 1000
        inp_size = self.inp_size

        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(2, inp_size - 1)),
            nn.Conv2d(10, 100, kernel_size=(2, 1)),
            nn.Conv2d(100, 100, kernel_size=(2, 1)),
            nn.Conv2d(100, 100, kernel_size=(2, 1)),
            nn.Conv2d(100, 100, kernel_size=(2, 1)),
            nn.Conv2d(100, 100, kernel_size=(2, 1)),
            nn.Conv2d(100, 100, kernel_size=(2, 1)),
            nn.Conv2d(100, 100, kernel_size=(2, 1)),
            nn.Conv2d(100, 100, kernel_size=(2, 1)),
            nn.Conv2d(100, 100, kernel_size=(2, 1)),
            nn.MaxPool2d((2, 1)))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, vocabulary_size),
            nn.Dropout(0.9))
        self.sf = nn.Softmax()

    def forward(self, x):
        emb = self.emb(x)
        out = self.features(emb)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.fc(out)
        out = self.sf(out)
        return out


class MLP(Model):

    def __init__(self, emb_size, win_size):
        super(MLP, self).__init__(emb_size, win_size)

        hidden_size = 512
        inp_size = self.inp_size

        self.layer1 = nn.Sequential(
            nn.Linear(2 * win_size * inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(hidden_size, vocabulary_size)
        self.sf = nn.Softmax()

    def forward(self, x):
        emb = self.emb(x)
        out = emb.view(emb.size(0), -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out)
        out = self.sf(out)
        return out


def device(device_str):
    return int(device_str[-1])


def predict(model, loader):
    model.eval()
    if use_cuda:
        with torch.cuda.device(device(args.device)):
            model.cuda()

    pred, truth, loss = [], [], []
    criterion, test_loss = nn.CrossEntropyLoss(), 0

    for i, (batch_inputs, batch_labels) in enumerate(loader):
        inp = torch.from_numpy(batch_inputs).long()
        lbl = torch.from_numpy(batch_labels).long().view(-1)

        if use_cuda:
            with torch.cuda.device(device(args.device)):
                inp = inp.cuda()
                lbl = lbl.cuda()

        x = Variable(inp)
        y = Variable(lbl)
        y_hat = model(x)
        loss.append(criterion(y_hat, y).data[0])
        pred.extend(y_hat.data.cpu().numpy().argmax(axis=1))
        truth.extend(y.data.cpu().numpy())

    return np.array(truth), np.array(pred), np.mean(loss)


def train(model, train_loader, test_loader):

    # Hyper Parameters
    num_epochs = args.num_epochs

    # Loss and Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    best_loss = np.inf
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '%s'" % args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])  # TODO: see if bug is fixed
            print("=> loaded checkpoint '%s' (epoch %s)" %
                  (args.resume, checkpoint['epoch'] + 1))
        else:
            print("=> no checkpoint found at '%s'" % args.resume)

    if use_cuda:
        with torch.cuda.device(device(args.device)):
            model.cuda()

    for epoch in range(start_epoch, num_epochs):

        for step, (batch_inputs, batch_labels) in enumerate(train_loader):

            inp = torch.from_numpy(batch_inputs).long()
            lbl = torch.from_numpy(batch_labels).long().view(-1)

            if use_cuda:
                with torch.cuda.device(device(args.device)):
                    inp = inp.cuda()
                    lbl = lbl.cuda()

            x = Variable(inp)
            y = Variable(lbl)

            model.train()
            optimizer.zero_grad()
            y_hat = model(x)

            loss = criterion(y_hat, y)

            train_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            # loss = p_pos.log() + p_neg.log()

            if (step + 1) % args.steps_per_stats == 0:

                truth, pred, test_loss = predict(model, test_loader)

                acc = np.sum(pred == truth) / truth.shape[0]

                f1 = f1_score(truth, pred, average='micro')

                print('Epoch [%d/%d], Train Loss: %.5f, Test Loss: %.5f, Test F1: %.2f, Test ACC: %.2f'
                      % (epoch + 1, num_epochs, train_loss / args.steps_per_stats, test_loss, f1, acc))
                train_loss = 0

                # remember best prec@1 and save checkpoint
                is_best = best_loss > test_loss
                best_loss = min(best_loss, test_loss)

                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict()
                }, is_best)


def save_checkpoint(state, is_best):
    filename_late = "%s/aapred_%s_latest.tar" % (ckptpath, arch)
    filename_best = "%s/aapred_%s_best.tar" % (ckptpath, arch)
    torch.save(state, filename_late)
    if is_best:
        copyfile(filename_late, filename_best)


def add_arguments(parser):
    parser.add_argument("-w", "--win_size", type=int, required=True,
                        help="Give the length of the context window.")
    parser.add_argument("-d", "--emb_dim", type=int, required=True,
                        help="Give the dimension of the embedding vector.")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Give the size of bach to use when training.")
    parser.add_argument("-e", "--num_epochs", type=int, default=10,
                        help="Give the number of epochs to use when training.")
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("-a", "--arch", type=str, choices=['mlp', 'cnn'],
                        default="mlp", help="Choose what type of model to use.")
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
    parser.add_argument("--device", type=str, default='cpu',
                        help="Specify what device you'd like to use e.g. 'cpu', 'gpu0' etc.")


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

    arch = args.arch

    use_cuda = args.device != 'cpu'

    ckptpath = args.out_dir
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)

    if arch == 'mlp':
        model = MLP(args.emb_dim, args.win_size)
        train_loader = WindowBatchLoader(args.win_size, args.batch_size)
        test_loader = WindowBatchLoader(args.win_size, args.batch_size, False)
        train(model, train_loader, test_loader)

    elif arch == 'cnn':
        model = CNN(args.emb_dim, args.win_size)
        train_loader = WindowBatchLoader(args.win_size, args.batch_size)
        test_loader = WindowBatchLoader(args.win_size, args.batch_size, False)
        train(model, train_loader, test_loader)

    else:
        print("Unknown model")
        exit(1)
