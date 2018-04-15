import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from src.python.digo_utils import *

from pymongo import MongoClient

from tqdm import tqdm

import numpy as np

from tempfile import gettempdir

import argparse

np.random.seed(101)

LR = 0.01

BATCH_SIZE = 32

LONG_EXPOSURE = True

USE_CUDA = True


def set_cuda(val):
    global USE_CUDA
    USE_CUDA = val


def get_loss(vec1, vec2, lbl, criterion=nn.CosineEmbeddingLoss()):
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

            nn.Conv1d(5, 1000, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(1000, 500, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(500, 500, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv1d(500, 500, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.embed = nn.Sequential(
            nn.Linear(500, 500),
            nn.Dropout(0.1)
        )

    def forward(self, seq):
        emb = self.embedding(seq)
        b = seq.size()[0]
        emb = emb.view((b, 5, -1))
        out = self.features(emb)
        out = F.max_pool1d(out, kernel_size=out.size()[2])
        out = out.view((b, -1))
        out = self.embed(out)
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


class MultiCosineLoss(nn.Module):
    def __init__(self):
        super(MultiCosineLoss, self).__init__()

    def forward(self, output1, target1):
        # cosine_target is used to evaluate the loss: 1-cos(y,y')

        cosine_target = Variable(torch.ones(len(output1)).cuda())
        loss_func = nn.CosineEmbeddingLoss().cuda()
        loss1 = loss_func(output1, target1, cosine_target)

        total_loss = loss1
        loss_func2 = torch.nn.L1Loss().cuda()
        output1_normed = output1.norm(p=2, dim=1)
        target = Variable(torch.ones(len(output1), 1)).cuda()
        loss21 = loss_func2(output1_normed, target)
        total_loss2 = loss21
        return 0.5 * total_loss + 0.5 * total_loss2


def batch_generator(data, labels, batch_size=BATCH_SIZE):

    def prepare_seq(sequence_obj, max_length=MAX_LENGTH):
        seq = sequence_obj.seq
        delta = max_length - len(seq)
        left = [PAD for _ in range(delta // 2)]
        right = [PAD for _ in range(delta - delta // 2)]
        seq = left + [AA.aa2index[aa] for aa in seq] + right
        return np.asarray(seq)

    def prepare_batch(seqs1, seqs2, labels):
        b1 = max(map(len, seqs1))
        b2 = max(map(len, seqs2))
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


def train(model, epoch, opt, gen_xy, length_xy):

    model.train()

    adjust_learning_rate(opt, epoch)

    pbar = tqdm(total=length_xy)

    err = 0

    for i, (seq1, seq2, lbl) in enumerate(gen_xy):

        opt.zero_grad()
        vec1 = model(seq1)
        vec2 = model(seq2)

        loss = get_loss(vec1, vec2, lbl)

        err += loss.data[0]
        loss.backward()
        opt.step()

        pbar.set_description("Training Loss:%.5f" % (err/(i + 1)))
        pbar.update(len(lbl))

    pbar.close()


def model_summary(model):
    for idx, m in enumerate(model.modules()):
        print(idx, '->', m)


def save_checkpoint(state, loss):
    filename_late = os.path.join(ckptpath, "digo_%.5f.tar" % loss)
    torch.save(state, filename_late)


def shuffle(data, labels):
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    return data[s], labels[s]


def adjust_learning_rate(optimizer, epoch):
    lr = LR * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def sample_pos_neg(graph, sample_size=15000):
#     pos, neg = set(), set()
#     pbar = tqdm(range(len(graph)), desc="nodes sampled")
#     for node in graph:
#         pbar.update(1)
#         if not node.is_leaf():
#             continue
#         s_in = min(10, node.size)
#         sample_in = np.random.choice(list(node.sequences), s_in, replace=False)
#         pos |= set((c1, c2, node.go) for c1, c2 in itertools.combinations(sample_in, 2))
#         for cousin in node.cousins:
#             s_out = min(10, cousin.size)
#             sample_out = np.random.choice(list(cousin.sequences), s_out, replace=False)
#             neg |= set((seq1, seq2, node.go) for seq1 in sample_out for seq2 in sample_in)
#     pbar.close()
#     n, m = len(pos), len(neg)
#     pos_indices = np.random.choice(list(range(n)), min(n, sample_size), replace=False)
#     neg_indices = np.random.choice(list(range(m)), min(m, sample_size), replace=False)
#     return np.asarray(list(pos))[pos_indices, :], np.asarray(list(neg))[neg_indices, :]


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
    parser.add_argument("--arch", type=str, choices=['deepseq', 'inception', 'motifnet'],
                        default="inception", help="Specify the model arch.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    asp = args.aspect  # default: Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    net = DeepSeq()

    ckptpath = args.out_dir

    if USE_CUDA: net = net.cuda()

    # opt = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    opt = optim.SGD(net.parameters(), lr=LR, momentum=0.9, nesterov=True)
    model_summary(net)

    print("#####################################################################")

    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prot2vec']

    asp = 'F'   # molecular function
    onto = get_ontology('F')

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

    size_trn = 50000
    size_tst = 10000
    pos_trn, neg_trn = sample_pos_neg(graph_trn, sample_size=size_trn)
    pos_tst, neg_tst = sample_pos_neg(graph_tst, sample_size=size_tst)

    lbl_trn = np.concatenate([np.ones(len(pos_trn)), -np.ones(len(neg_trn))])
    data_trn = np.concatenate([pos_trn, neg_trn], axis=0)
    lbl_tst = np.concatenate([np.ones(len(pos_tst)), -np.ones(len(neg_tst))])
    data_tst = np.concatenate([pos_tst, neg_tst], axis=0)

    data_trn, lbl_trn = shuffle(data_trn, lbl_trn)
    data_tst, lbl_tst = shuffle(data_tst, lbl_tst)

    print("Train: %d, Test: %d" % (len(data_trn), len(data_tst)))

    num_epochs = 200

    for epoch in range(args.init_epoch, num_epochs):

        train(net, epoch + 1, opt, batch_generator(data_trn, lbl_trn), size_trn * 2)

        if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
            continue

        loss = evaluate(net, batch_generator(data_tst, lbl_tst), size_tst * 2)

        print("[Epoch %d/%d] (Validation Loss: %.5f" % (epoch + 1, num_epochs, loss))

        save_checkpoint({
            'epoch': epoch,
            'net': net.state_dict(),
            'opt': opt.state_dict()
        }, loss)
