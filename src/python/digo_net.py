import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from src.python.digo_utils import *

from src.python.pytorch_utils import *

from tqdm import tqdm

import numpy as np

from tempfile import gettempdir

import argparse

np.random.seed(101)

LR = 0.01

LEARN_GO = False

ATTN = "general"

BATCH_SIZE = 32

USE_CUDA = True

VERBOSE = True


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
    for i, (seq1, seq2, node, lbl) in enumerate(gen_xy):
        vec1 = net(seq1, node)
        vec2 = net(seq2, node)
        loss = get_loss(vec1, vec2, lbl)
        err += loss.data[0]
        pbar.set_description("Validation Loss:%.5f" % (err/(i + 1)))
        pbar.update(len(lbl))
    pbar.close()
    return err / (i + 1)


class Go2Vec(nn.Module):
    def __init__(self, emb_weights, requires_grad=LEARN_GO):
        super(Go2Vec, self).__init__()
        embedding_size = emb_weights.shape[1]
        self.embedding = nn.Embedding(emb_weights.shape[0], embedding_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(emb_weights).float())
        self.embedding.requires_grad = requires_grad

    def forward(self, go):
        return self.embedding(go)


class ProteinEncoder(nn.Module):
    def __init__(self, num_channels, embedding_size, dropout=0.1):
        super(ProteinEncoder, self).__init__()

        self.aa_embedding = nn.Embedding(26, embedding_size)

        self.cnn = nn.Sequential(

            nn.Conv1d(5, num_channels * 2, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(num_channels * 2, num_channels, kernel_size=15),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.embedding_size = embedding_size
        self.dropout = dropout

    def forward(self, input_seqs, hidden=None):
        embedded_seqs = self.aa_embedding(input_seqs)
        input_features = self.cnn(embedded_seqs.transpose(1, 2))
        return input_features


class AttnDecoder(nn.Module):

    def __init__(self, attn_model, num_channels, prot_section_size, go_embedding_weights, dropout=0.1):
        super(AttnDecoder, self).__init__()

        self.prot_encoder = ProteinEncoder(num_channels, 5)

        self.dropout = dropout

        # Keep for reference
        self.attn_model = attn_model
        self.prot_section_size = prot_section_size
        self.num_channels = num_channels
        self.dropout = dropout

        # Define layers
        self.go_embedding_size = go_embedding_size = go_embedding_weights.shape[1]
        self.go_embedding = Go2Vec(go_embedding_weights)
        self.embedding_dropout = nn.Dropout(dropout)
        self.attn = Attn(attn_model, num_channels * prot_section_size, go_embedding_size)

    def forward(self, input_seq, input_go_term):
        encoder_outputs = self.prot_encoder(input_seq)
        encoder_outputs = self.embedding_dropout(encoder_outputs)
        batch_size = encoder_outputs.size(0)
        protein_length = encoder_outputs.size(2)
        prot_section_size = self.prot_section_size
        new_prot_length = protein_length // prot_section_size
        remainder = protein_length % prot_section_size
        head = remainder // 2
        tail = protein_length - (remainder - head)
        encoder_outputs = encoder_outputs[:, :, head:tail].contiguous()
        encoder_outputs = encoder_outputs.view(batch_size, -1, new_prot_length)
        embedded_go = self.go_embedding(input_go_term)
        attn_weights = self.attn(embedded_go, encoder_outputs)
        context_vec = attn_weights.bmm(encoder_outputs.transpose(1, 2))
        return context_vec.squeeze(1)


class Attn(nn.Module):
    def __init__(self, method, protein_section_size, go_embedding_size):
        super(Attn, self).__init__()

        self.method = method
        self.go_size = go_embedding_size
        self.prot_size = protein_section_size

        if self.method == 'general':
            self.attn = nn.Linear(self.prot_size, self.go_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.go_size + self.prot_size, self.go_size)
            self.v = nn.Parameter(torch.FloatTensor(1, self.go_size))

        else:
            raise ValueError("unknown attn method")

    def forward(self, go_embedding, protein_sections):
        max_len = protein_sections.size(2)
        this_batch_size = protein_sections.size(0)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

            for i in range(max_len):
                batch_protein_sections = protein_sections[:, :, i]
                attn_energies[:, i] = self.score(go_embedding, batch_protein_sections)

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, go_embedding, protein_section):

        go_size = self.go_size
        prot_size = self.prot_size
        batch_size = go_embedding.size(0)

        if self.method == 'dot':
            assert prot_size == go_size
            energy = torch.bmm(go_embedding.view(batch_size, 1, go_size),
                               protein_section.view(batch_size, prot_size, 1))
            return energy

        elif self.method == 'general':
            energy = self.attn(protein_section)
            energy = torch.bmm(go_embedding.view(batch_size, 1, go_size),
                               energy.view(batch_size, go_size, 1))
            return energy

        elif self.method == 'concat':
            energy = F.tanh(self.attn(torch.cat((go_embedding, protein_section), 1)))
            print(self.v.size(), energy.size())
            energy = torch.bmm(self.v.view(batch_size, 1, go_size),
                               energy.view(1, go_size, 1))
            return energy


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

    def prepare_node(node):
        return onto.classes.index(node.go)

    def prepare_batch(seqs1, seqs2, nodes, labels, extra_padding=10):
        b1 = max(map(len, seqs1)) + extra_padding
        b2 = max(map(len, seqs2)) + extra_padding
        inp1 = np.asarray([prepare_seq(seq, b1) for seq in seqs1])
        inp2 = np.asarray([prepare_seq(seq, b2) for seq in seqs2])
        inp3 = np.asarray([prepare_node(node) for node in nodes])
        inp_var1 = Variable(torch.LongTensor(inp1))
        inp_var2 = Variable(torch.LongTensor(inp2))
        inp_var3 = Variable(torch.LongTensor(inp3))
        lbl_var = Variable(torch.FloatTensor(labels))
        if USE_CUDA:
            inp_var1 = inp_var1.cuda()
            inp_var2 = inp_var2.cuda()
            inp_var3 = inp_var3.cuda()
            lbl_var = lbl_var.cuda()
        return inp_var1, inp_var2, inp_var3, lbl_var

    indices = list(range(0, len(data), batch_size))
    np.random.shuffle(indices)
    while indices:
        ix = indices.pop()
        batch_inp = data[ix: min(ix + batch_size, len(data))]
        lbls = labels[ix: min(ix + batch_size, len(labels))]
        seqs1, seqs2, nodes = zip(*batch_inp)
        yield prepare_batch(seqs1, seqs2, nodes, lbls)


def train(model, opt, adalr, gen_xy, length_xy):

    model.train()

    pbar = tqdm(total=length_xy)

    err = 0

    for i, (seq1, seq2, node, lbl) in enumerate(gen_xy):

        opt.zero_grad()
        vec1 = model(seq1, node)
        vec2 = model(seq2, node)

        loss = get_loss(vec1, vec2, lbl)
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
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="How many epochs to train the model?")
    parser.add_argument("--arch", type=str, choices=['deepseq', 'inception', 'motifnet'],
                        default="inception", help="Specify the model arch.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument('-p', '--pos_only', action='store_true', default=False,
                        help="Whether to train on positive pairs only?")


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

    size_trn = 200000
    size_tst = 10000

    if args.pos_only:
        data_trn = sample_pairs(graph_trn.leaves, True, size_trn)
        lbl_trn = np.ones(len(data_trn))
        data_tst = sample_pairs(graph_tst.leaves, True, size_tst)
        lbl_tst = np.ones(len(data_tst))
    else:
        pos_trn, neg_trn = sample_pos_neg(graph_trn, sample_size=size_trn)
        data_trn = np.concatenate([pos_trn, neg_trn], axis=0)
        lbl_trn = np.concatenate([np.ones(len(pos_trn)), -np.ones(len(neg_trn))])
        pos_tst, neg_tst = sample_pos_neg(graph_tst, sample_size=size_tst)
        data_tst = np.concatenate([pos_tst, neg_tst], axis=0)
        lbl_tst = np.concatenate([np.ones(len(pos_tst)), -np.ones(len(neg_tst))])

    size_trn = len(data_trn)
    size_tst = len(data_tst)

    data_trn, lbl_trn = shuffle(data_trn, lbl_trn)
    data_tst, lbl_tst = shuffle(data_tst, lbl_tst)

    print("|Train|: %d, |Test|: %d, Batch_Size: %d, Learn_GO: %s"
          % (len(data_trn), len(data_tst), BATCH_SIZE, LEARN_GO))

    print("#####################################################################")

    go_embedding_weights = np.asarray([onto.todense(go) for go in onto.classes])

    net = AttnDecoder(ATTN, 100, 10, go_embedding_weights)

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
            if 'lr' in checkpoint: LR = checkpoint['lr']
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

        train(net, opt, adalr, batch_generator(data_trn, lbl_trn), size_trn)

        if epoch < num_epochs - 1 and epoch % args.eval_every != 0:
            continue

        loss = evaluate(net, batch_generator(data_tst, lbl_tst), size_tst)

        if VERBOSE:
            print("[Epoch %d/%d] (Validation Loss: %.5f" % (epoch + 1, num_epochs, loss))

        save_checkpoint({
            'lr': adalr.lr,
            'epoch': epoch,
            'net': net.state_dict(),
            'opt': opt.state_dict()
        }, loss, "digo", ckptpath)
