# import os
# import numpy as np
#
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
#
# from gensim.models.word2vec import Word2Vec
#
# from pymongo import MongoClient
#
# from tempfile import gettempdir
#
# from obonet import read_obo
#
# from src.python.preprocess import *
#
# import argparse
#
#
# # CNN Model (2 conv layer)
# class CNN_AC(nn.Module):
#     def __init__(self, n_classes):
#         super(CNN_AC, self).__init__()
#         self.n_classes = n_classes
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.fc = nn.Linear(8 * 5 * 32, n_classes)
#         self.sf = nn.Softmax()
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         out = self.sf(out)
#         return out
#
#
# def train(model, train_dataset, test_dataset, output_dir):
#
#     print("Training Model: #TRN=%d , #VLD=%d, #CLS=%d" %
#           (len(train_dataset), len(test_dataset), model.n_classes))
#
#     # Hyper Parameters
#
#     batch_size = 32
#     num_epochs = 5
#     learning_rate = 0.003
#
#     # Loss and Optimizer
#     criterion = nn.MultiLabelMarginLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     for epoch in range(num_epochs):
#
#         # pbar = tqdm(range(len(train_dataset)), "training ... ")
#
#         loader = DataLoader(train_dataset, batch_size)
#         for i, (seqs, lbls) in enumerate(loader):
#
#             model.train()
#             sequences = Variable(torch.from_numpy(seqs).float())
#             labels = Variable(torch.from_numpy(lbls).long())
#
#             # Forward + Backward + Optimize
#             optimizer.zero_grad()
#             outputs = model(sequences)
#
#             train_loss = criterion(outputs, labels)
#             train_loss.backward()
#             optimizer.step()
#
#             if (i + 1) % 100 == 0:
#
#                 test_loss = eval(model, criterion, test_dataset)
#                 print('Epoch [%d/%d], Step [%d/%d], Train Loss: %.4f, Test Loss: %.4f'
#                                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
#                                         train_loss.data[0], test_loss))
#             # pbar.update(batch_size)
#
#         # pbar.close()
#
#         # Save the Trained Model
#         torch.save(model.state_dict(), '%s/cnn.pkl' % output_dir)
#
#
# def eval(model, criterion, dataset):
#     model.eval()
#     batch_size = 32
#     loss = 0.0
#     loader = DataLoader(dataset, batch_size)
#     for i, (seqs, lbls) in enumerate(loader):
#         sequences = Variable(torch.from_numpy(seqs).float())
#         labels = Variable(torch.from_numpy(lbls).long())
#         outputs = model(sequences)
#         loss += criterion(outputs, labels).data[0]
#     return loss
#
#
# def main():
#
#     output_dir = args.out_dir
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#
#     client = MongoClient("mongodb://127.0.0.1:27017")
#     db = client['prot2vec']
#
#     exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"]
#     queries = [{'Evidence': {'$in': exp_codes}, 'Aspect': asp, 'DB': 'UniProtKB'} for asp in ['F', 'C', 'P']]
#
#     num_mfo, num_cco, num_bpo = [db.goa_uniprot.count(q) for q in queries]
#     src_mfo, src_cco, src_bpo = [db.goa_uniprot.find(q).limit(10000) for q in queries]
#     # src_mfo, src_cco, src_bpo = [db.goa_uniprot.find(q) for q in queries]
#
#     print("#ANNO MF=%s CC=%s BP=%s" % (num_mfo, num_cco, num_bpo))
#     seqid2goid, goid2seqid = GoAnnotationCollectionLoader(src_mfo, 10000, 'F').load()
#     # seqid2goid, goid2seqid = GoAnnotationCollectionLoader(src_mfo, num_mfo, 'F').load()
#
#     query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
#     num_seq = db.uniprot.count(query)
#     src_seq = db.uniprot.find(query)
#     print("#SEQ=%s" % num_seq)
#     seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()
#
#     dataset = Dataset(seqid2seq, seqid2goid)
#     print(dataset)
#     p99 = np.percentile(list(map(lambda r: len(r.seq), dataset.records)), 99)
#     print("99 percentile:\t%d" % p99)
#
#     print("Reading GeneOntology")
#     GO = read_obo("http://purl.obolibrary.org/obo/go/go-basic.obo")
#
#     dataset.augment(GO)
#     print(dataset)
#
#     train_set, test_set = dataset.split(0.2)
#
#
#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("-o", "--out_dir", type=str, required=False,
#                         default=gettempdir(), help="Specify the output directory.")
#     parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
#                         help="Supply the URL of MongoDB")
#     parser.add_argument("--device", type=str, default='cpu',
#                         help="Specify what device you'd like to use e.g. 'cpu', 'gpu0' etc.")
#
#     args = parser.parse_args()
#
#     main()
