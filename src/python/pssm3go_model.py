import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


USE_CUDA = False

KERN_SIZE = 3


def set_cuda(val):
    global USE_CUDA
    USE_CUDA = val


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    # if sequence_length.is_cuda:
    if USE_CUDA:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length, gamma=0, eps=1e-7):
    length = Variable(torch.LongTensor(length))
    if USE_CUDA:
        length = length.cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # probs_flat: (batch * max_len, 1)
    probs_flat = torch.gather(F.softmax(logits_flat), dim=1, index=target_flat)
    probs_flat = probs_flat.clamp(eps, 1. - eps)   # prob: [0, 1] -> [eps, 1 - eps]
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.log(probs_flat) * (1 - probs_flat) ** gamma  # focal loss
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()

        inp_size = input_size

        self.features = nn.Sequential(

            nn.Conv2d(1, 10, kernel_size=(KERN_SIZE, inp_size)),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),

            nn.Conv2d(10, 10, kernel_size=(KERN_SIZE, 1)),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2, 1)),

            nn.Conv2d(10, 20, kernel_size=(KERN_SIZE, 1)),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),

            nn.Conv2d(20, 20, kernel_size=(KERN_SIZE, 1)),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2, 1)),

            nn.Conv2d(20, 40, kernel_size=(KERN_SIZE, 1)),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),

            nn.Conv2d(40, 40, kernel_size=(KERN_SIZE, 1)),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((2, 1)),
        )

        self.n_pool_layers = 3

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(2), out.size(0), out.size(1) * out.size(3))
        return out


class EncoderCNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderCNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.cnn = CNN(input_size)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        input_features = self.cnn(input_seqs.transpose(0, 1).unsqueeze(1))
        features_length = [(l//(2 ** self.cnn.n_pool_layers)) for l in input_lengths]
        # features_length = input_lengths
        # print(input_features.size())
        # print(features_length)
        # Note: we run this all at once (over multiple batches of multiple sequences)
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_features, features_length)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy


class LuongAttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, prior_size=0, dropout=0.1, embedding=None):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.prior_size = prior_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        if np.any(embedding):
            self.embedding_size = embedding_size = embedding.shape[1]
            self.embedding = nn.Embedding(output_size, embedding_size)
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding).float())
            self.embedding.requires_grad = True
        else:
            self.embedding_size = embedding_size = hidden_size
            self.embedding = nn.Embedding(output_size, embedding_size)

        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size + prior_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, prior):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, -1)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        if prior is None:
            output = self.out(concat_output)
        else:
            output = self.out(torch.cat((concat_output, prior), 1))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()
