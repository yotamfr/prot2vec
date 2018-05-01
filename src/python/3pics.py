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

from src.python.pssm2go_model import *

from src.python.baselines import *

from src.python.consts import *

from pymongo import MongoClient

from tempfile import gettempdir

from shutil import copyfile

import pickle

import argparse

verbose = True

ckptpath = gettempdir()

SHOW_PLOT = False

USE_CUDA = False

PAD_token = 0
SOS_token = 1
EOS_token = 2


MIN_LENGTH = 50
MAX_LENGTH = 500

MIN_COUNT = 2

GAP = '-'

t0 = datetime.datetime(2016, 2, 1, 0, 0)
t1 = datetime.datetime(2017, 12, 1, 0, 0)


set_verbose(False)


def labeled_3pics(db, query, t, limit):

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)
    seqid2goid, _ = GoAnnotationCollectionLoader(s, c, ASPECT).load()

    q = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.pssm.count(q)
    src_seq = db.pssm.find(q)

    seqid2seqpssm = PssmCollectionLoader(src_seq, num_seq).load()
    seqid2seqpssm = {k: v for k, v in seqid2seqpssm.items() if len(seqid2seqpssm[k][2]) > 1}
    seqid2goid = {k: v for k, v in seqid2goid.items() if k in seqid2seqpssm}

    ids, pics1, pics2, pics3 = [], [], [], []
    for i, (seqid, (seq, pssm, msa)) in enumerate(seqid2seqpssm.items()):
        # sys.stdout.write("\r{0:.0f}%".format(100.0 * i / len(seqid2seqpssm)))
        query["DB_Object_ID"] = {"$in": [r[0].split('|')[1] for r in msa[1:]]}
        homoids = [r[0].split('|')[1] for r in msa[1:]]
        homodocs = db.goa_uniprot.find({"DB": "UniProtKB",
                                        "DB_Object_ID": {"$in": homoids},
                                        "Date": {"$lte": t}})
        homologos = {}
        for i, doc in len(homodocs):
            k, v = doc["DB_Object_ID"], doc["GO_ID"]
            if k in homologos:
                homologos[k].append(v)
            else:
                homologos[k] = [v]

        gomap = [[k, homologos[k] if k in homologos else []] for k in homoids]
        print(i)
        print(len(homologos))
        print(len(gomap))

        pics1.append(profile2pic(pssm, seq))
        pics2.append(msa2pic(msa))
        pics3.append(gomap2pic(gomap))
        ids.append(seqid)

    labels = [seqid2goid[k] for k in ids]

    return ids, pics1, pics2, pics3, labels


def gomap2pic(gomap):
    global onto
    return []


def msa2pic(msa):
    return [[-1. if aa == GAP else AA.aa2index[aa] for aa in aln] for _, aln in msa]


def profile2pic(pssm, seq):
    return [AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)]
            for i, aa in enumerate(seq)]


def load_training_and_validation(db, limit=None):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$lte": t0},
               'Aspect': ASPECT}

    trn_ids, trn_pics1, trn_pics2, trn_pics3, trn_labels = labeled_3pics(db, q_train, t0, None)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$gt": t0, "$lte": t1},
               'Aspect': ASPECT}

    tst_ids, tst_pics1, tst_pics2, tst_pics3, tst_labels = labeled_3pics(db, q_valid, t1, limit)

    return trn_ids, trn_pics1, trn_pics2, trn_pics3, trn_labels, tst_ids, tst_pics1, tst_pics2, tst_pics3, tst_labels


def filter_pairs(pairs_gen):
    filtered_pairs = []
    original_pairs = []
    for _, inp, out in pairs_gen:
        original_pairs.append((inp, out))
        if MIN_LENGTH <= len(inp) <= MAX_LENGTH:
            filtered_pairs.append((inp, out))
    return original_pairs, filtered_pairs


class PssmGoPairsGen(object):

    def __init__(self, seqid2seqpssm, seqid2goid):

        self.seqid2seqpssm = seqid2seqpssm
        self.seqid2goid = seqid2goid

    def __iter__(self):
        seqid2seqpssm = self.seqid2seqpssm
        seqid2goid = self.seqid2goid
        sorted_keys = sorted(seqid2goid.keys(), key=lambda k: len(seqid2seqpssm[k][0]))
        for seqid in sorted_keys:
            annots = seqid2goid[seqid]
            seq, pssm, msa = seqid2seqpssm[seqid]
            if len(pssm) != len(seq) or len(msa) == 1:
                print("WARN: wrong PSSM! (%s)" % seqid)
                continue
            for head, seq in msa[1:]:
                _, seqid, _ = head.split('|')
                annots = map(lambda doc: doc[""], db.goa_uniprot.f)
            matrix = [AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)]
                      for i, aa in enumerate(seq)]
            sent_go = onto.propagate(annots, include_root=False)
            yield (seqid, matrix, sent_go)


def prepare_data(pairs_gen):

    pairs1, pairs2 = filter_pairs(pairs_gen)
    print("Filtered %d to %d pairs" % (len(pairs1), len(pairs2)))

    print("Indexing words...")
    for pair in pairs2:
        output_lang.index_words(pair[1])

    print('Indexed %d words in GO' % output_lang.n_words)
    return pairs2


def trim_pairs(pairs):
    keep_pairs, trimmed_pairs = [], []

    for i, pair in enumerate(pairs):

        n = len(pairs)

        if verbose:
            sys.stdout.write("\r{0:.0f}%".format(100.0 * i / n))

        input_seq, output_annots = pair
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
    seq += [PAD_token for _ in range(max_length - len(seq))]
    return seq


def random_batch(batch_size):

    # Choose random pairs
    ix = random.choice(list(range(len(pairs)-batch_size)))
    input_seqs = sorted([pair[0] for pair in pairs[ix:ix+batch_size]], key=lambda s: -len(s))
    target_seqs = [indexes_from_sequence(output_lang, pair[1]) for pair in pairs[ix:ix+batch_size]]

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

    return input_var, input_lengths, target_var, target_lengths


def test_models():

    small_batch_size = 3
    input_batches, input_lengths, target_batches, target_lengths = random_batch(small_batch_size)

    print('input_batches', input_batches.size())  # (max_len x batch_size)
    print('target_batches', target_batches.size())  # (max_len x batch_size)

    small_hidden_size = 8
    small_n_layers = 2

    encoder_test = EncoderRNN(input_size, small_hidden_size, small_n_layers)
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


def train(input_batches, input_lengths, target_batches, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          batch_size, grad_clip, gamma):
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


def evaluate(encoder, decoder, input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_batches = Variable(torch.FloatTensor([input_seq]), volatile=True).transpose(0, 1)

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


def evaluate_and_show_attention(encoder, decoder, input_seq, target_words=None):
    output_words, attentions = evaluate(encoder, decoder, input_seq)
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
                        default="F", help="Specify the ontology aspect.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument("-m", "--model_name", type=str, required=False,
                        default="pssm2go", help="Specify the model name.")
    parser.add_argument("-q", '--quiet', action='store_true', default=False,
                        help="Run in quiet mode.")
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help="Specify whether to use pretrained embeddings.")
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
    batch_size=12,
    # batch_size=50,

    # Configure training/optimization
    clip=50.0,
    gamma=1.0,
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

    # Initialize models
    if not args.cnn:
        encoder = EncoderRNN(input_size, encoder_hidden_size, n_layers, dropout=dropout)
    else:
        encoder = EncoderRCNN(input_size, encoder_hidden_size, n_layers, dropout=dropout)

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
        input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer,
            batch_size, clip, gamma
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

    data = load_training_and_validation(db, limit=1000)
    pass

    # input_size = len(AA) * 2
    #
    # gen = PssmGoPairsGen(seqid2seqpssm, seqid2goid)
    # pairs = prepare_data(gen)
    #
    # output_lang.trim(MIN_COUNT)
    #
    # save_object(output_lang, os.path.join(ckptpath, "go-lang-%s.pkl" % GoAspect(args.aspect)))
    #
    # pairs, _ = trim_pairs(pairs)
    #
    # test_models()
    #
    # if args.pretrained:
    #     output_embedding = np.array([onto.todense(go) for go
    #                                  in sorted(output_lang.word2index.keys(),
    #                                            key=lambda k: output_lang.word2index[k])])
    #     dummy_embedding = np.random.rand(3, output_embedding.shape[1])
    #     output_embedding = np.concatenate((dummy_embedding, output_embedding))
    # else:
    #     input_embedding = None
    #     output_embedding = None
    #
    # main_loop(
    #     print_every=args.print_every,
    #     evaluate_every=args.eval_every
    # )
