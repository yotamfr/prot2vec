import os
import datetime
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.preprocessing import MultiLabelBinarizer

from torch.autograd import Variable

from gensim.models.word2vec import Word2Vec

from pymongo import MongoClient

from tqdm import tqdm

from preprocess import *

import argparse


cafa2_cutoff = datetime.datetime(2014, 1, 1, 0, 0)
cafa3_cutoff = datetime.datetime(2017, 2, 2, 0, 0)
today_cutoff = datetime.datetime.now()

cafa3_targets_url = 'http://biofunctionprediction.org/cafa-targets/CAFA3_targets.tgz'
cafa3_train_url = 'http://biofunctionprediction.org/cafa-targets/CAFA3_training_data.tgz'
cafa2_data_url = 'https://ndownloader.figshare.com/files/3658395'
cafa2_targets_url = 'http://biofunctionprediction.org/cafa-targets/CAFA-2013-targets.tgz'


# CNN Model (2 conv layer)
class CNN_AC(nn.Module):
    def __init__(self, n_classes):
        super(CNN_AC, self).__init__()
        self.n_classes = n_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(8 * 5 * 32, n_classes)
        self.sf = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sf(out)
        return out


class ToAC(object):

    def __init__(self, lag):
        self.lag = lag

    def __call__(self, seq):
        n, J = seq.shape
        Lag = self.lag
        X = seq-np.sum(seq, axis=0)
        X_t = X.transpose()
        AC = np.zeros((J, Lag))
        for lag in range(1, Lag):
            tmp = np.divide(np.dot(X_t[:, :(n - lag)], X[lag:, :]), n-lag)
            AC[:, lag] = np.diagonal(tmp)
        return AC


def train(model, train_dataset, test_dataset, output_dir):

    print("Training Model: #TRN=%d , #VLD=%d, #CLS=%d" %
          (len(train_dataset), len(test_dataset), model.n_classes))

    # Hyper Parameters

    batch_size = 32
    num_epochs = 5
    learning_rate = 0.003

    # Loss and Optimizer
    criterion = nn.MultiLabelMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        pbar = tqdm(range(len(train_dataset)), "training ... ")

        loader = DataLoader(train_dataset, batch_size)
        for i, (seqs, lbls) in enumerate(loader):

            model.train()
            sequences = Variable(torch.from_numpy(seqs).float())
            labels = Variable(torch.from_numpy(lbls).long())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(sequences)

            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:

                test_loss = eval(model, criterion, test_dataset)
                pbar.set_description('Epoch [%d/%d], Step [%d/%d], Train Loss: %.4f, Test Loss: %.4f'
                                     % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                                        train_loss.data[0], test_loss))
            pbar.update(batch_size)

        pbar.close()

        # Save the Trained Model
        torch.save(model.state_dict(), '%s/cnn.pkl' % output_dir)


def eval(model, criterion, dataset):
    model.eval()
    batch_size = 32
    loss = 0.0
    loader = DataLoader(dataset, batch_size)
    for i, (seqs, lbls) in enumerate(loader):
        sequences = Variable(torch.from_numpy(seqs).float())
        labels = Variable(torch.from_numpy(lbls).long())
        outputs = model(sequences)
        loss += criterion(outputs, labels).data[0]
    return loss


def main():

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']

    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    seq2vec = Seq2Vec(Word2Vec.load(args.model))

    if args.source == 'CAFA3':

        cafa3_train_dir = '%s/CAFA3_training_data' % input_dir
        if not os.path.exists(cafa3_train_dir):
            wget_and_unzip('CAFA3_training_data', input_dir, cafa3_train_url)

        cafa3_go_tsv = '%s/%s/uniprot_sprot_exp.txt' % (input_dir, cafa3_train_dir)
        cafa3_train_fasta = '%s/%s/uniprot_sprot_exp.fasta' % (input_dir, cafa3_train_dir)
        seq_id2seq, seq_id2go_id, go_id2seq_id = \
            load_training_data_from_files(cafa3_go_tsv, cafa3_train_fasta, GoAspect('F'))
        filter_sequences_by(lambda seq: len(seq) < 32, seq_id2seq, seq_id2go_id)
        train_set = Dataset(seq_id2seq, seq2vec, seq_id2go_id, transform=ToAC(32))

        cafa3_targets_dir = '%s/Target files' % input_dir
        cafa3_mapping_dir = '%s/Mapping files' % input_dir
        if not os.path.exists(cafa3_targets_dir) or not os.path.exists(cafa3_mapping_dir):
            wget_and_unzip('CAFA3_targets', input_dir, cafa3_targets_url)

        annots_fname = 'leafonly_MFO_unique.txt'
        annots_fpath = '%s/CAFA3_benchmark20170605/groundtruth/%s' % (input_dir, annots_fname)
        trg_id2seq, _, _ = load_cafa3_targets(cafa3_targets_dir, cafa3_mapping_dir)
        num_mapping = count_lines(annots_fpath, sep=bytes('\n', 'utf8'))
        src_mapping = open(annots_fpath, 'r')
        trg_id2go_id, go_id2trg_id = MappingFileLoader(src_mapping, num_mapping, annots_fname).load()
        filter_sequences_by(lambda seq: len(seq) < 32, trg_id2seq, trg_id2go_id)
        test_set = Dataset(trg_id2seq, seq2vec, trg_id2go_id, transform=ToAC(32))

        seq_id2seq, seq_id2go_id, go_id2seq_id = \
            load_training_data_from_collections(db.goa_uniprot, db.uniprot,
                                                cafa3_cutoff, today_cutoff, GoAspect('F'))
        filter_sequences_by(lambda seq: len(seq) < 32, seq_id2seq, seq_id2go_id)
        valid_set = Dataset(seq_id2seq, seq2vec, seq_id2go_id, transform=ToAC(32))

        all_labels = []
        all_labels.extend(train_set.labels)
        all_labels.extend(valid_set.labels)
        all_labels.extend(test_set.labels)
        mlb = MultiLabelBinarizer(sparse_output=False).fit(all_labels)
        train_set.mlb, valid_set.mlb, test_set.mlb = mlb, mlb, mlb
        print(mlb.classes_)

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        cnn = CNN_AC(len(mlb.classes_))
        train(cnn, train_set, test_set, output_dir)

    elif args.source == 'CAFA2':

        sub_dir = cafa2_targets_dir = 'CAFA-2013-targets'
        if not os.path.exists('%s/%s' % (input_dir, sub_dir)):
            wget_and_unzip(sub_dir, input_dir, cafa2_targets_url)
        sub_dir = cafa2_data_dir = 'CAFA2Supplementary_data'
        if not os.path.exists('%s/%s' % (input_dir, sub_dir)):
            wget_and_unzip(sub_dir, input_dir, cafa2_data_url)

        cafa2_targets_dir = './CAFA2Supplementary_data/data/CAFA2-targets'
        cafa2_benchmark_dir = './CAFA2Supplementary_data/data/benchmark'

    else:
        print('Unrecognized source')
        exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Supply input directory for the raw data")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Supply output directory for the processed data")
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("-s", "--source", type=str, required=True,
                        choices=['CAFA3', 'CAFA2'],
                        help="Specify the source of the data that you wish to load.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Specify the path to the embedding model.")
    args = parser.parse_args()

    main()
