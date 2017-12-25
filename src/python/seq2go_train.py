import os
import sys

import random
import time

import math

from shutil import copyfile

import torchvision
from torch import optim

import io

from PIL import Image
import visdom
vis = visdom.Visdom()

import sconce

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import socket
hostname = socket.gethostname()

from .preprocess import *

from .seq2go_model import *

from .embedding2 import *

from pymongo import MongoClient

from tempfile import gettempdir

import argparse

verbose = True

ckptpath = gettempdir()

SHOW_PLOT = False

USE_CUDA = False

KMER = 3

PAD_token = 0
SOS_token = 1
EOS_token = 2

# MIN_LENGTH = 3
# MAX_LENGTH = 25
MIN_LENGTH = 1
MAX_LENGTH = 500

MIN_COUNT = 5
# MIN_COUNT = 2


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


def load_data(db, asp, codes=exp_codes, limit=None):
    q = {'Evidence': {'$in': codes}, 'DB': 'UniProtKB'}
    c = limit if limit else db.goa_uniprot.count(q)
    s = db.goa_uniprot.find(q)
    if limit: s = s.limit(limit)

    seqid2goid, goid2seqid = GoAnnotationCollectionLoader(s, c, asp).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    return seqid2seq, goid2seqid, seqid2goid


def filter_pairs(pairs_gen):
    filtered_pairs = []
    original_pairs = []
    for pair in pairs_gen:
        original_pairs.append(pair)
        if MIN_LENGTH <= len(pair[0]) <= MAX_LENGTH and MIN_LENGTH <= len(pair[1]) <= MAX_LENGTH:
            filtered_pairs.append(pair)
    return original_pairs, filtered_pairs


class KmerGoPairsGen(object):

    def __init__(self, kmer):
        self.k = kmer

    def __iter__(self):

        for (seqid, annots) in seqid2goid.items():
            seq = seqid2seq[seqid]
            sent_go = onto.sort(onto.augment(annots))
            for offset in range(self.k):
                sent_kmer = get_kmer_sentences(seq, self.k, offset)
                if not np.all([(w in kmer_w2v) for w in sent_kmer]):
                    continue
                yield (sent_kmer, sent_go)


def prepare_data(pairs_gen):

    pairs1, pairs2 = filter_pairs(pairs_gen)
    print("Filtered %d to %d pairs" % (len(pairs1), len(pairs2)))

    print("Indexing words...")
    for pair in pairs2:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    return pairs2


def trim_pairs(pairs):
    keep_pairs = []

    for i, pair in enumerate(pairs):

        n = len(pairs)

        if verbose:
            sys.stdout.write("\r{0:.0f}%".format(100.0 * i / n))

        input_seq = pair[0]
        output_annots = pair[1]
        keep_input = True
        keep_output = True

        for word in input_seq:
            if word not in input_lang.word2index:
                keep_input = False
                break

        for word in output_annots:
            if word not in output_lang.word2index:
                keep_output = False
                break

        # Remove if pair doesn't match input and output conditions
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("\nTrimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Return a list of indexes, one for each word in the sequence, plus EOS
def indexes_from_sequence(lang, seq):
    return [lang.word2index[word] for word in seq] + [EOS_token]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def random_batch(batch_size):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sequence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sequence(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths


def test_models():

    small_batch_size = 3
    input_batches, input_lengths, target_batches, target_lengths = random_batch(small_batch_size)

    print('input_batches', input_batches.size())  # (max_len x batch_size)
    print('target_batches', target_batches.size())  # (max_len x batch_size)

    small_hidden_size = 8
    small_n_layers = 2

    encoder_test = EncoderRNN(input_lang.n_words, small_hidden_size, small_n_layers)
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
            decoder_input, decoder_hidden, encoder_outputs
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


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, batch_size, grad_clip):
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
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
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


def evaluate(encoder, decoder, input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [indexes_from_sequence(input_lang, input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

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
            decoder_input, decoder_hidden, encoder_outputs
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
    [input_seq, target_seq] = random.choice(pairs)
    evaluate_and_show_attention(encoder, decoder, input_seq, target_seq)


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


def evaluate_and_show_attention(encoder, decoder, input_words, target_words=None):
    output_words, attentions = evaluate(encoder, decoder, input_words)
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
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument("-q", '--quiet', action='store_true', default=False,
                        help="Run in quiet mode.")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-d", "--device", type=str, default='cpu',
                        help="Specify what device you'd like to use e.g. 'cpu', 'gpu0' etc.")
    parser.add_argument("-p", "--print_every", type=int, default=1,
                        help="How often should main_loop print training stats.")
    parser.add_argument("-e", "--eval_every", type=int, default=10,
                        help="How often should main_loop evaluate the model.")
    parser.add_argument("-l", "--max_length", type=int, default=500,
                        help="Max sequence length (both input and output).")
    parser.add_argument("-c", "--min_count", type=int, default=5,
                        help="Minimal word count (both input and output).")


def save_checkpoint(state, is_best=False):
    filename_late = "%s/seq2go_latest.tar" % ckptpath
    torch.save(state, filename_late)
    if is_best:
        filename_best = "%s/seq2go_best.tar" % ckptpath
        copyfile(filename_late, filename_best)


def main_loop(
    # Configure models
    attn_model='general',
    decoder_hidden_size=1000,
    encoder_hidden_size=1000,
    n_layers=2,
    dropout=0.1,
    batch_size=50,

    # Configure training/optimization
    clip=50.0,
    teacher_forcing_ratio=0.5,
    learning_rate=0.0001,
    decoder_learning_ratio=5.0,
    n_epochs=50000,
    epoch=0,
    plot_every=20,
    print_every=20,
    evaluate_every=1000
):
    assert encoder_hidden_size == decoder_hidden_size

    input_embedding = np.array([kmer_w2v[kmer] for kmer
                                in sorted(input_lang.word2index.keys(),
                                          key=lambda k: input_lang.word2index[k])])
    output_embedding = np.array([onto.todense(go) for go
                                 in sorted(output_lang.word2index.keys(),
                                           key=lambda k: output_lang.word2index[k])])

    # Initialize models
    encoder = EncoderRNN(input_lang.n_words, encoder_hidden_size, n_layers,
                         dropout=dropout, embedding=input_embedding)
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

    job = sconce.Job('seq2go-nmt', {
        'attn_model': attn_model,
        'n_layers': n_layers,
        'dropout': dropout,
        'decoder_hidden_size': decoder_hidden_size,
        'encoder_hidden_size': encoder_hidden_size,
        'learning_rate': learning_rate,
        'clip': clip,
        'teacher_forcing_ratio': teacher_forcing_ratio,
        'decoder_learning_ratio': decoder_learning_ratio,
    })
    job.plot_every = plot_every
    job.log_every = print_every

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
        input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer,
            batch_size, clip
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc

        job.record(epoch, loss)

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


if __name__ == "__main__":

    # Load and Prepare the data
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    USE_CUDA = 'gpu' in args.device
    set_cuda(USE_CUDA)

    MAX_LENGTH = args.max_length
    MIN_COUNT = args.min_count

    if USE_CUDA:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]

    verbose = not args.quiet

    ckptpath = args.out_dir

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']

    onto = get_ontology('F')
    
    stream = map(lambda p: p['sequence'], db.uniprot.find({'db': 'sp'}))
    kmer_w2v = Word2VecWrapper("3mer", KmerSentencesLoader(3, list(stream)))

    seqid2seq, goid2seqid, seqid2goid = load_data(db, 'F', limit=None)

    input_lang = Lang("KMER")
    output_lang = Lang("GO")
    gen = KmerGoPairsGen(KMER)
    pairs = prepare_data(gen)

    input_lang.trim(MIN_COUNT)
    output_lang.trim(MIN_COUNT)

    pairs = trim_pairs(pairs)

    test_models()

    main_loop(print_every=args.print_every, evaluate_every=args.eval_every)
