import os
import numpy as np

import torch
import torch.nn as nn


from torch.autograd import Variable

from gensim.models.word2vec import Word2Vec

from pymongo import MongoClient

from tqdm import tqdm

from preprocess import *

import argparse


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

        # pbar = tqdm(range(len(train_dataset)), "training ... ")

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
                print('Epoch [%d/%d], Step [%d/%d], Train Loss: %.4f, Test Loss: %.4f'
                                     % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                                        train_loss.data[0], test_loss))
            # pbar.update(batch_size)

        # pbar.close()

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

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.source == 'CAFA3':

        train_set, valid_set, test_set, mlb = \
            load_cafa3(db, input_dir, 'F', Word2Vec.load(args.model),
                       lambda seq: len(seq) < 32, lambda seqs: len(seqs) < 5,
                       trans=ToAC(32))

        print("Splitting Dataset into Train and Test.")

        data_set = train_set.update(valid_set).update(test_set)

        train_set, valid_set = data_set.split()

        cnn = CNN_AC(len(mlb.classes_))
        train(cnn, train_set, valid_set, output_dir)

    elif args.source == 'CAFA2':

        train_set, valid_set, test_set, mlb = \
            load_cafa2(db, input_dir, Word2Vec.load(args.model),
                       lambda seq: len(seq) < 32, lambda seqs: len(seqs) < 5,
                       trans=ToAC(32))

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
