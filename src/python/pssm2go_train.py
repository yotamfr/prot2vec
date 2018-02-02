import os
import sys

import random
import time

import math

import torchvision
from torch import optim

import io

from PIL import Image
import visdom
vis = visdom.Visdom()

import matplotlib.ticker as ticker

import socket
hostname = socket.gethostname()

from .pssm2go_model import *

from .baselines import *

from .consts import *

from pymongo import MongoClient

from tempfile import gettempdir

from shutil import copyfile

import pickle

import argparse

verbose = True

ckptpath = gettempdir()

SHOW_PLOT = False

USE_CUDA = False

USE_PRIOR = False

PAD_token = 0
SOS_token = 1
EOS_token = 2

MIN_LENGTH = 48
MAX_LENGTH = 480

MIN_COUNT = 2

t0 = datetime(2016, 2, 1, 0, 0)
t1 = datetime(2017, 2, 1, 0, 0)


class Lang(object):
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

    def index_words(self, sequence):
        for word in sequence:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

        for word in keep_words:
            self.index_word(word)


def _get_labeled_data(db, query, limit, pssm=True):

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    seqid2goid, _ = GoAnnotationCollectionLoader(s, c, ASPECT).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}

    if pssm:
        num_seq = db.pssm.count(query)
        src_seq = db.pssm.find(query)
        seqid2seq = PssmCollectionLoader(src_seq, num_seq).load()
    else:
        num_seq = db.uniprot.count(query)
        src_seq = db.uniprot.find(query)
        seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    seqid2goid = {k: v for k, v in seqid2goid.items() if len(v) > 1 or 'GO:0005515' not in v}
    seqid2goid = {k: v for k, v in seqid2goid.items() if k in seqid2seq.keys()}

    return seqid2seq, seqid2goid


def load_training_and_validation(db, limit=None):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$lte": t0},
               'Aspect': ASPECT}

    sequences_train, annotations_train = _get_labeled_data(db, q_train, limit)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$gt": t0, "$lte": t1},
               'Aspect': ASPECT}

    sequences_valid, annotations_valid = _get_labeled_data(db, q_valid, limit)
    forbidden = set(sequences_train.keys())
    sequences_valid = {k: v for k, v in sequences_valid.items() if k not in forbidden}
    annotations_valid = {k: v for k, v in annotations_valid.items() if k not in forbidden}

    return sequences_train, annotations_train, sequences_valid, annotations_valid


def filter_pairs(pairs_gen):
    filtered_pairs = []
    original_pairs = []
    for _, inp, prior, out in pairs_gen:
        original_pairs.append((inp, prior, out))
        if MIN_LENGTH <= len(inp) <= MAX_LENGTH:
            filtered_pairs.append((inp, prior, out))
    return original_pairs, filtered_pairs


class DataGenerator(object):

    def __init__(self, seqid2seqpssm, seqid2goid, blast2go=None, one_leaf=False):

        self.seqid2seqpssm = seqid2seqpssm
        self.seqid2goid = seqid2goid
        self.blast2go = blast2go
        self.one_leaf = one_leaf

    def __iter__(self):
        seqid2goid = self.seqid2goid
        seqid2seqpssm = self.seqid2seqpssm
        for seqid in sorted(seqid2goid.keys(), key=lambda k: len(seqid2seqpssm[k][0])):
            seq, pssm, _ = seqid2seqpssm[seqid]

            if len(pssm) != len(seq):
                print("WARN: wrong PSSM! (%s)" % seqid)
                continue

            matrix = [AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)]
                      for i, aa in enumerate(seq)]

            if self.blast2go:
                prior = self.blast2go[seqid]
            else:
                prior = None

            if self.one_leaf:
                annots = []
                for leaf in seqid2goid[seqid]:
                    anc = onto.propagate([leaf], include_root=False)
                    if len(anc) > len(annots):
                        annots = anc
            else:
                annots = onto.propagate(seqid2goid[seqid], include_root=False)

            yield (seqid, matrix, prior, annots)


def prepare_data(pairs_gen):

    pairs1, pairs2 = filter_pairs(pairs_gen)
    print("Filtered %d to %d pairs" % (len(pairs1), len(pairs2)))

    print("Indexing words...")
    for pair in pairs2:
        output_lang.index_words(pair[2])

    print('Indexed %d words in GO' % output_lang.n_words)
    return pairs2


def trim_pairs(pairs):
    keep_pairs, trimmed_pairs = [], []

    for i, pair in enumerate(pairs):

        n = len(pairs)

        if verbose:
            sys.stdout.write("\r{0:.0f}%".format(100.0 * i / n))

        input_seq, prior_annots, output_annots = pair
        keep_input = True
        keep_output = True

        for word in output_annots:
            if word not in output_lang.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)
        else:
            trimmed_pairs.append(pair)

    print("\nTrimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs, trimmed_pairs


# Return a list of indexes, one for each word in the sequence, plus EOS
def indexes_from_sequence(lang, seq):
    return [lang.word2index[word] for word in seq] + [EOS_token]


# Pad a with zeros
def pad_inp(seq, max_length):
    seq = [(seq[i] if i < len(seq) else ([0.] * input_size)) for i in range(max_length)]
    return seq


# Pad a with the PAD symbol
def pad_out(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size):

    # Choose random pairs
    ix = random.choice(list(range(len(trn_pairs) - batch_size)))
    sample = sorted([x for x in trn_pairs[ix:ix + batch_size]], key=lambda x: -len(x[0]))
    input_seqs = [inp for (inp, _, _) in sample]
    if USE_PRIOR:
        input_prior = [[prior[go] if go in prior else 0. for go in output_lang.word2index.keys()] for (_, prior, _) in sample]
        prior_var = Variable(torch.FloatTensor(input_prior))
    else:
        prior_var = None

    target_seqs = [indexes_from_sequence(output_lang, out) for (_, _, out) in sample]

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_inp(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_out(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.FloatTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
    if USE_CUDA and USE_PRIOR:
        prior_var = prior_var.cuda()

    return input_var, input_lengths, target_var, target_lengths, prior_var


def test_models():

    small_batch_size = 3
    input_batches, input_lengths, target_batches, target_lengths, input_prior = \
        random_batch(small_batch_size)

    print('input_batches', input_batches.size())  # (max_len x batch_size)
    print('target_batches', target_batches.size())  # (max_len x batch_size)

    small_hidden_size = 8
    small_n_layers = 2

    encoder_test = EncoderRNN(input_size, small_hidden_size, small_n_layers)
    if USE_PRIOR:
        decoder_test = LuongAttnDecoderRNN('general', small_hidden_size, output_lang.n_words, small_n_layers, output_lang.n_words - 3)
    else:
        decoder_test = LuongAttnDecoderRNN('general', small_hidden_size, output_lang.n_words, small_n_layers)

    if USE_CUDA:
        encoder_test.cuda()
        decoder_test.cuda()

    encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths, None)

    print('encoder_outputs', encoder_outputs.size())  # max_len x batch_size x hidden_size
    print('encoder_hidden', encoder_hidden.size())  # n_layers * 2 x batch_size x hidden_size

    max_target_length = max(target_lengths)

    # Prepare decoder input and outputs
    decoder_input = Variable(torch.LongTensor([SOS_token] * small_batch_size))
    decoder_hidden = encoder_hidden[:decoder_test.n_layers] # Use last (forward) hidden state from encoder
    all_decoder_outputs = Variable(torch.zeros(max_target_length, small_batch_size, decoder_test.output_size))

    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()
        decoder_input = decoder_input.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder_test(
            decoder_input, decoder_hidden, encoder_outputs, input_prior
        )
        all_decoder_outputs[t] = decoder_output # Store this step's outputs
        decoder_input = target_batches[t] # Next input is current target

    # Test masked cross entropy loss
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths
    )
    print('loss', loss.data[0])


def train(input_batches, input_lengths, target_batches, target_lengths, input_prior,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          batch_size, grad_clip, gamma, teacher_forcing):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs, input_prior
        )

        all_decoder_outputs[t] = decoder_output
        if teacher_forcing == 1:
            decoder_input = target_batches[t]  # Next input is current target
        else:
            # Choose top word from output
            _, topi = decoder_output.data.topk(1)
            decoder_input = Variable(topi.squeeze(1))

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths, gamma=gamma
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def evaluate(encoder, decoder, input_seq, prior=None, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_batches = Variable(torch.FloatTensor([input_seq]), volatile=True).transpose(0, 1)
    if USE_PRIOR:
        input_prior = [prior[go] if go in prior else 0. for go in output_lang.word2index.keys()]
        input_prior = Variable(torch.FloatTensor([input_prior]))
    else:
        input_prior = None

    if USE_CUDA:
        input_batches = input_batches.cuda()
    if USE_CUDA and USE_PRIOR:
        input_prior = input_prior.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs, input_prior
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate_randomly(encoder, decoder):
    [input_seq, prior, target_seq] = random.choice(tst_pairs)
    evaluate_and_show_attention(encoder, decoder, input_seq, target_seq, prior)


def show_attention(input_sequence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sequence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    show_plot_visdom()
    plt.show()
    plt.close()


def show_plot_visdom():
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    im = Image.open(buf).convert("RGB")
    vis.image(torchvision.transforms.ToTensor()(im), win=attn_win, opts={'title': attn_win})


def evaluate_and_show_attention(encoder, decoder, input_seq, target_words=None, input_prior=None):
    output_words, attentions = evaluate(encoder, decoder, input_seq, input_prior)
    input_words = [AA.index2aa[vec[:len(AA)].index(1) if 1 in vec[:len(AA)] else 20]
                   for vec in input_seq]
    output_sequence = ' '.join(output_words)
    input_sequence = ' '.join(input_words)
    target_sequence = ' '.join(target_words)
    print('>', input_sequence)
    if target_sequence is not None:
        print('=', target_sequence)
    print('<', output_sequence)

    if not SHOW_PLOT:
        return

    show_attention(input_sequence, output_words, attentions)

    # Show input, target, output text in visdom
    win = 'evaluted (%s)' % hostname
    text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sequence, target_sequence, output_sequence)
    vis.text(text, win=win, opts={'title': win})


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument('--cnn', action='store_true', default=False,
                        help="Use CNN to extract features from input sequence.")
    parser.add_argument("-a", "--aspect", type=str, choices=['F', 'P', 'C'],
                        required=True, help="Specify the ontology aspect.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument("-m", "--model_name", type=str, required=False,
                        default="pssm2go", help="Specify the model name.")
    parser.add_argument("-q", '--quiet', action='store_true', default=False,
                        help="Run in quiet mode.")
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help="Specify whether to use pretrained embeddings.")
    parser.add_argument('--blast2go', action='store_true', default=False,
                        help="Specify whether to use blast2go predictions.")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-d", "--device", type=str, default='cpu',
                        help="Specify what device you'd like to use e.g. 'cpu', 'gpu0' etc.")
    parser.add_argument("-p", "--print_every", type=int, default=10,
                        help="How often should main_loop print training stats.")
    parser.add_argument("-e", "--eval_every", type=int, default=100,
                        help="How often should main_loop evaluate the model.")
    parser.add_argument("-l", "--max_length", type=int, default=200,
                        help="Max sequence length (both input and output).")
    parser.add_argument("-c", "--min_count", type=int, default=2,
                        help="Minimal word count (both input and output).")
    parser.add_argument("--num_cpu", type=int, default=4,
                        help="How many cpus for computing blast2go prior")


def save_checkpoint(state, is_best=False):
    filename_late = os.path.join(ckptpath, "%s-%s-latest.tar"
                                 % (args.model_name, GoAspect(args.aspect)))
    torch.save(state, filename_late)
    if is_best:
        filename_best = os.path.join(ckptpath, "%s-%s-best.tar"
                                     % (args.model_name, GoAspect(args.aspect)))
        copyfile(filename_late, filename_best)


# https://github.com/pytorch/pytorch/issues/2830
def optimizer_cuda(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


def main_loop(
    # Configure models
    attn_model='general',
    decoder_hidden_size=500,
    encoder_hidden_size=500,
    n_layers=2,
    dropout=0.1,
    batch_size=6,
    # batch_size=50,

    # Configure training/optimization
    clip=50.0,
    gamma=2.0,
    teacher_forcing_ratio=0.8,
    learning_rate=0.0001,
    decoder_learning_ratio=5.0,
    n_epochs=50000,
    epoch=0,
    plot_every=20,
    print_every=20,
    evaluate_every=1000
):
    assert encoder_hidden_size == decoder_hidden_size

    # Initialize models
    if not args.cnn:
        encoder = EncoderRNN(input_size, encoder_hidden_size, n_layers, dropout=dropout)
    else:
        encoder = EncoderCNN(input_size, encoder_hidden_size, n_layers, dropout=dropout)

    if USE_PRIOR:
        decoder = LuongAttnDecoderRNN(attn_model, decoder_hidden_size, output_lang.n_words, n_layers,
                                      dropout=dropout, embedding=output_embedding, prior_size=output_lang.n_words - 3)
    else:
        decoder = LuongAttnDecoderRNN(attn_model, decoder_hidden_size, output_lang.n_words, n_layers,
                                      dropout=dropout, embedding=output_embedding)

    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '%s'" % args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            epoch = checkpoint['epoch']
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        else:
            print("=> no checkpoint found at '%s'" % args.resume)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()
    if USE_CUDA and args.resume:
        optimizer_cuda(encoder_optimizer)
        optimizer_cuda(decoder_optimizer)

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Begin!
    ecs = []
    dcs = []
    eca = 0
    dca = 0

    while epoch < n_epochs:
        epoch += 1

        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths, input_prior =\
            random_batch(batch_size)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths, input_prior,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer,
            batch_size, clip, gamma,
            np.random.binomial(1, teacher_forcing_ratio)
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc

        # job.record(epoch, loss)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % evaluate_every == 0:
            evaluate_randomly(encoder, decoder)

            save_checkpoint({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict()
                })

        if not SHOW_PLOT:
            continue

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            # TODO: Running average helper
            ecs.append(eca / plot_every)
            dcs.append(dca / plot_every)
            ecs_win = 'encoder grad (%s)' % hostname
            dcs_win = 'decoder grad (%s)' % hostname
            vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
            vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
            eca = 0
            dca = 0


def set_output_lang(lang):
    global output_lang
    output_lang = lang


def set_ontology(ontology):
    global onto
    onto = ontology


def set_show_attn(val):
    global SHOW_PLOT
    SHOW_PLOT = val


def set_use_cuda(val):
    global USE_CUDA
    USE_CUDA = val
    set_cuda(val)


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    # Load and Prepare the data
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    set_use_cuda('gpu' in args.device)

    MAX_LENGTH = args.max_length
    MIN_COUNT = args.min_count

    if USE_CUDA:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]

    verbose = not args.quiet

    ckptpath = args.out_dir

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']

    onto = init_GO(args.aspect)

    trn_seq2pssm, trn_seq2go, tst_seq2pssm, tst_seq2go = load_training_and_validation(db, limit=None)

    if args.blast2go:
        USE_PRIOR = True
        pred_path = os.path.join(tmp_dir, 'pred-blast-%s.npy' % GoAspect(ASPECT))
        if os.path.exists(pred_path):
            blast2go = np.load(pred_path).item()
        else:
            targets = {k: v[0] for k, v in list(trn_seq2pssm.items()) + list(tst_seq2pssm.items())}
            q = {'DB': 'UniProtKB', 'Evidence': {'$in': exp_codes}, 'Date':  {"$lte": t0}, 'Aspect': ASPECT}
            reference, _ = _get_labeled_data(db, q, limit=None, pssm=False)
            blast2go = parallel_blast(targets, reference, num_cpu=args.num_cpu)
            np.save(pred_path, blast2go)
    else:
        USE_PRIOR = False
        blast2go = None

    input_size = len(AA) * 2

    output_lang = Lang("GO")
    trn_pairs = prepare_data(DataGenerator(trn_seq2pssm, trn_seq2go, blast2go))
    tst_pairs = prepare_data(DataGenerator(tst_seq2pssm, tst_seq2go, blast2go))

    output_lang.trim(MIN_COUNT)

    save_object(output_lang, os.path.join(ckptpath, "go-lang-%s.pkl" % GoAspect(args.aspect)))

    trn_pairs, _ = trim_pairs(trn_pairs)
    tst_pairs, _ = trim_pairs(tst_pairs)

    test_models()

    if args.pretrained:
        output_embedding = np.array([onto.todense(go) for go
                                     in sorted(output_lang.word2index.keys(),
                                               key=lambda k: output_lang.word2index[k])])
        dummy_embedding = np.random.rand(3, output_embedding.shape[1])
        output_embedding = np.concatenate((dummy_embedding, output_embedding))
    else:
        input_embedding = None
        output_embedding = None

    main_loop(
        print_every=args.print_every,
        evaluate_every=args.eval_every
    )
