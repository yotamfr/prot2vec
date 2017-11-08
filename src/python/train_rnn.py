import os
import sys
import datetime


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
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

cafa3_benchmark_dir = './CAFA3_benchmark20170605/groundtruth'


# BiRNN Model (Many-to-One)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))  # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


def train(dataset):

    print("Training RNN...\n")

    # Hyper Parameters
    # sequence_length = 28
    input_size = dataset[0][0].shape[1]
    hidden_size = 512
    num_layers = 4
    num_classes = len(dataset.classes)
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.003

    rnn = BiRNN(input_size, hidden_size, num_layers, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        pbar = tqdm(range(len(dataset)), "records trained")

        for i, (seqs, lbls) in enumerate(DataLoader(dataset, batch_size)):

            print(seqs.shape)
            print(lbls.shape)

            sequences = Variable(torch.from_numpy(seqs).float())
            labels = Variable(torch.from_numpy(lbls).float())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(dataset) // batch_size, loss.data[0]))

            pbar.update(batch_size)

        pbar.close()


def eval(dataset):
    pass


def main():

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']

    data_dir = args.input_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    seq2vec = Seq2Vec(Word2Vec.load(args.model))

    if args.source == 'CAFA3':

        sub_dir = cafa3_train_dir = 'CAFA3_training_data'
        if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
            wget_and_unzip(sub_dir, data_dir, cafa3_train_url)
        # sub_dir = cafa3_targets_dir = 'CAFA3_targets'
        # if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
        #     wget_and_unzip(sub_dir, data_dir, cafa3_targets_url)

        # cafa3_targets_dir = '%s/Target files' % data_dir
        # cafa3_mapping_dir = '%s/Mapping files' % data_dir
        # load_cafa3_targets(cafa3_targets_dir, cafa3_mapping_dir)

        cafa3_go_tsv = '%s/%s/uniprot_sprot_exp.txt' % (data_dir, cafa3_train_dir)
        cafa3_train_fasta = '%s/%s/uniprot_sprot_exp.fasta' % (data_dir, cafa3_train_dir)
        seq_id2seq, seq_id2go_id, go_id2seq_id = \
            load_training_data_from_files(cafa3_go_tsv, cafa3_train_fasta, GoAspect('F'))

        train_dataset = Dataset(seq_id2seq, seq2vec, seq_id2go_id)
        train(train_dataset)

        seq_id2seq, seq_id2go_id, go_id2seq_id = \
            load_training_data_from_collections(db.goa_uniprot, db.uniprot,
                                                cafa3_cutoff, today_cutoff, GoAspect('F'))

        train(train_dataset)

        eval_dataset = Dataset(seq_id2seq, seq2vec, seq_id2go_id)
        eval(eval_dataset)


    elif args.source == 'CAFA2':

        sub_dir = cafa2_targets_dir = 'CAFA-2013-targets'
        if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
            wget_and_unzip(sub_dir, data_dir, cafa2_targets_url)
        sub_dir = cafa2_data_dir = 'CAFA2Supplementary_data'
        if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
            wget_and_unzip(sub_dir, data_dir, cafa2_data_url)

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
