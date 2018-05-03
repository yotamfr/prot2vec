# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from src.python.dingo_utils2 import *

from src.python.preprocess2 import *

from src.python.dingo_sampling import *

from src.python.pytorch_utils import *

from src.python.consts import *

from tqdm import tqdm

import numpy as np

from tempfile import gettempdir

import argparse

import datetime

np.random.seed(101)

LR = 0.1

LEARN_GO = False

ATTN = "self"

BATCH_SIZE = 20

USE_CUDA = True

VERBOSE = True


def set_cuda(val):
    global USE_CUDA
    USE_CUDA = val


def get_loss(vec1, vec2, lbl):
    loss = criterion(vec1.float(), vec2.float(), lbl.float())
    return loss


def evaluate(model, gen_xy, length_xy):
    model.eval()
    pbar = tqdm(total=length_xy)
    err = 0
    for i, (seq1, seq2, lbl) in enumerate(gen_xy):
        vec1 = net(seq1)
        vec2 = net(seq2)
        loss = get_loss(vec1, vec2, lbl)
        err += loss.data[0]
        pbar.set_description("Validation Loss:%.5f" % (err/(i + 1)))
        pbar.update(len(lbl))
    pbar.close()
    return err / (i + 1)


class DeepSeq(nn.Module):
    def __init__(self):
        super(DeepSeq, self).__init__()

        self.embedding = nn.Embedding(26, 5)

        self.features = nn.Sequential(

            nn.Conv1d(5, 1000, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(1000, 500, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(500, 500, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(500, 500, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, seq):
        emb = self.embedding(seq)
        b = seq.size()[0]
        emb = emb.view((b, 5, -1))
        out = self.features(emb)
        out = F.max_pool1d(out, kernel_size=out.size()[2])
        out = out.view((b, -1))
        return out


def prepare_seq(sequence_obj, max_length=MAX_LENGTH):
    seq = sequence_obj.seq
    delta = max_length - len(seq)
    left = [PAD for _ in range(delta // 2)]
    right = [PAD for _ in range(delta - delta // 2)]
    seq = left + [AA.aa2index[aa] for aa in seq] + right
    return np.asarray(seq)


def prepare_node(node, onto):
    return onto.classes.index(node.go)


def pairs_generator(data, labels, batch_size=BATCH_SIZE):

    def prepare_batch(seqs1, seqs2, labels, extra_padding=10):
        b1 = max(map(len, seqs1)) + extra_padding
        b2 = max(map(len, seqs2)) + extra_padding
        inp1 = np.asarray([prepare_seq(seq, b1) for seq in seqs1])
        inp2 = np.asarray([prepare_seq(seq, b2) for seq in seqs2])
        inp_var1 = Variable(torch.LongTensor(inp1))
        inp_var2 = Variable(torch.LongTensor(inp2))
        lbl_var = Variable(torch.FloatTensor(labels))
        if USE_CUDA:
            inp_var1 = inp_var1.cuda()
            inp_var2 = inp_var2.cuda()
            lbl_var = lbl_var.cuda()
        return inp_var1, inp_var2, lbl_var

    indices = list(range(0, len(data), batch_size))
    np.random.shuffle(indices)
    while indices:
        ix = indices.pop()
        batch_inp = data[ix: min(ix + batch_size, len(data))]
        lbls = labels[ix: min(ix + batch_size, len(labels))]
        seqs1, seqs2 = zip(*batch_inp)
        yield prepare_batch(seqs1, seqs2, lbls)


def compute_vectors(data, model, onto, batch_size=BATCH_SIZE):

    model.eval()

    def prepare_batch(seqs, nodes, extra_padding=10):
        b = max(map(len, seqs)) + extra_padding
        inp_seq = np.asarray([prepare_seq(seq, b) for seq in seqs])
        inp_node = np.asarray([prepare_node(node, onto) for node in nodes])
        var_seq = Variable(torch.LongTensor(inp_seq))
        var_node = Variable(torch.LongTensor(inp_node))
        if USE_CUDA:
            var_seq = var_seq.cuda()
            var_node = var_node.cuda()
        return var_seq, var_node

    pbar = tqdm(range(len(data)), desc="records processed")
    indices = list(range(0, len(data), batch_size))
    while indices:
        ix = indices.pop()
        batch_inp = data[ix: min(ix + batch_size, len(data))]
        seqs, nodes = zip(*batch_inp)
        var_seq, var_node = prepare_batch(seqs, nodes)
        var_vec = model(var_seq, var_node)
        vecs = var_vec.data.cpu().numpy()
        for seq, node, vec in zip(seqs, nodes, vecs):
            node.seq2vec[seq] = vec
        pbar.update(len(vecs))
    pbar.close()


def train(model, training_manager, gen_xy, length_xy):

    model.train()

    opt = training_manager.opt

    pbar = tqdm(total=length_xy)

    err = 0

    for i, (seq1, seq2, lbl) in enumerate(gen_xy):

        opt.zero_grad()
        vec1 = model(seq1)
        vec2 = model(seq2)

        # print(vec1)
        # print(vec2)

        loss = get_loss(vec1, vec2, lbl)
        # print(loss.data[0])
        adalr.update(loss.data[0])

        err += loss.data[0]
        loss.backward()
        opt.step()

        pbar.set_description("Training Loss:%.5f" % (err/(i + 1)))
        pbar.update(len(lbl))

    pbar.close()


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB"),
    parser.add_argument("--aspect", type=str, choices=['F', 'P', 'C'],
                        default="F", help="Specify the ontology aspect.")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-e", "--eval_every", type=int, default=1,
                        help="How often to evaluate on the validation set.")
    parser.add_argument("-n", "--num_epochs", type=int, default=100,
                        help="How many epochs to train the model?")
    parser.add_argument("-s",  "--sample", type=str, choices=['cousins', 'pos_only', 'no_common', 'iou'],
                        default="no_common", help="Specify the sampling technique.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    # parser.add_argument('-p', '--pos_only', action='store_true', default=False,
    #                     help="Whether to train on positive pairs only?")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    asp = args.aspect  # default: Molecular Function
    onto = get_ontology(asp)

    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prot2vec']

    t0 = datetime.datetime(2014, 1, 1, 0, 0)
    t1 = datetime.datetime(2014, 9, 1, 0, 0)
    # t0 = datetime.datetime(2017, 1, 1, 0, 0)
    # t1 = datetime.datetime.utcnow()

    print("Indexing Data...")
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, asp)
    print("Loading Training Data...")
    uid2seq_trn, _, go2ids_trn = trn_stream.to_dictionaries(propagate=True)
    print("Loading Validation Data...")
    uid2seq_tst, _, go2ids_tst = tst_stream.to_dictionaries(propagate=True)

    print("Building Train Graph...")
    graph_trn = Graph(onto, uid2seq_trn, go2ids_trn)
    print("Graph contains %d nodes" % len(graph_trn))

    print("Building Validation Graph...")
    graph_tst = Graph(onto, uid2seq_tst, go2ids_tst)
    print("Graph contains %d nodes" % len(graph_tst))

    size_trn = 300000
    size_tst = 10000

    # criterion = nn.CosineEmbeddingLoss(margin=0.1)
    criterion = CosineSimilarityLossWithL2Regularization()

    if args.sample == "pos_only":
        data_trn = sample_pairs(graph_trn.leaves, sample_size=size_trn)
        lbl_trn = np.ones(len(data_trn))
        data_tst = sample_pairs(graph_tst.leaves, sample_size=size_tst)
        lbl_tst = np.ones(len(data_tst))
    elif args.sample == "cousins":
        pos_trn, neg_trn = sample_pos_neg(graph_trn, sample_size=size_trn)
        data_trn = np.concatenate([pos_trn, neg_trn], axis=0)
        lbl_trn = np.concatenate([np.ones(len(pos_trn)), -np.ones(len(neg_trn))])
        pos_tst, neg_tst = sample_pos_neg(graph_tst, sample_size=size_tst)
        data_tst = np.concatenate([pos_tst, neg_tst], axis=0)
        lbl_tst = np.concatenate([np.ones(len(pos_tst)), -np.ones(len(neg_tst))])
    elif args.sample == "no_common":
        pos_trn, neg_trn = sample_pos_neg_no_common_ancestors(graph_trn, sample_size=size_trn)
        data_trn = np.concatenate([pos_trn, neg_trn], axis=0)
        lbl_trn = np.concatenate([np.ones(len(pos_trn)), -np.ones(len(neg_trn))])
        pos_tst, neg_tst = sample_pos_neg_no_common_ancestors(graph_tst, sample_size=size_tst)
        data_tst = np.concatenate([pos_tst, neg_tst], axis=0)
        lbl_tst = np.concatenate([np.ones(len(pos_tst)), -np.ones(len(neg_tst))])
    # elif args.sample == "iou":
    #     seqs1, seqs2, labels = zip(*sample_pairs_iou(graph_trn, sample_size=size_trn))
    #     data_trn, lbl_trn = np.asarray(list(zip(seqs1, seqs2))), np.asarray(labels)
    #     seqs1, seqs2, labels = zip(*sample_pairs_iou(graph_tst, sample_size=size_tst))
    #     data_tst, lbl_tst = np.asarray(list(zip(seqs1, seqs2))), np.asarray(labels)
    #     criterion = CosineSimilarityRegressionLoss()
    else:
        print("Unrecognized sampling technique")
        exit(0)

    size_trn = len(data_trn)
    size_tst = len(data_tst)

    data_trn, lbl_trn = shuffle(data_trn, lbl_trn)
    data_tst, lbl_tst = shuffle(data_tst, lbl_tst)

    print("|Train|: %d, |Test|: %d, Batch_Size: %d, Learn_GO: %s"
          % (len(data_trn), len(data_tst), BATCH_SIZE, LEARN_GO))

    print("#####################################################################")

    net = DeepSeq()

    ckptpath = args.out_dir

    opt = optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)

    model_summary(net)

    init_epoch = 0
    num_epochs = args.num_epochs

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '%s'" % args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            # if 'lr' in checkpoint: LR = checkpoint['lr']
            init_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            opt.load_state_dict(checkpoint['opt'])
        else:
            print("=> no checkpoint found at '%s'" % args.resume)

    adalr = AdaptiveLR(opt, LR)

    # Move models to GPU
    if USE_CUDA:
        net = net.cuda()
    if USE_CUDA and args.resume:
        optimizer_cuda(opt)

    for epoch in range(init_epoch, num_epochs):

        train(net, adalr, pairs_generator(data_trn, lbl_trn), size_trn)

        if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
            continue

        loss = evaluate(net, pairs_generator(data_tst, lbl_tst), size_tst)

        if VERBOSE:
            print("[Epoch %d/%d] (Validation Loss: %.5f" % (epoch + 1, num_epochs, loss))

        save_checkpoint({
            'lr': adalr.lr,
            'epoch': epoch,
            'net': net.state_dict(),
            'opt': opt.state_dict(),
        }, loss, "dingo", ckptpath)
