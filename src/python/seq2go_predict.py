import sys

from pymongo import MongoClient

from .seq2go_train import *
from .baselines import *
import numpy as np


def init_encoder_decoder(input_vocab_size,
                         output_vocab_size,
                         input_embedding=None,
                         output_embedding=None,
                         encoder_hidden_size=500,
                         decoder_hidden_size=500,
                         attn_model="general",
                         n_layers=2, dropout=0.1):
    enc = EncoderRNN(input_vocab_size, encoder_hidden_size, n_layers,
                     dropout=dropout, embedding=input_embedding)
    dec = LuongAttnDecoderRNN(attn_model, decoder_hidden_size, output_vocab_size, n_layers,
                              dropout=dropout, embedding=output_embedding)

    return enc, dec


def load_encoder_decoder_weights(encoder, decoder, resume_path):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '%s'" % resume_path)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        # epoch = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
    else:
        print("=> no checkpoint found at '%s'" % args.resume)


def predict(encoder, decoder, input_seq, max_length=MAX_LENGTH):
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
    decoded_words, annotations = [], {}
    attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        for ix, prob in enumerate(decoder_output.data):
            if ix == EOS_token:
                break
            go = output_lang.index2word[ix]
            if go in annotations:
                annotations[go].append(prob)
            else:
                annotations[go] = [prob]

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

    for go, ps in annotations.items():
        annotations[go] = 1 - np.prod([(1 - p) for p in ps])

    return decoded_words, attentions[:di + 1, :len(encoder_outputs)], annotations


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("-a", "--aspect", type=str, choices=['F', 'P', 'C'],
                        required=True, help="Specify the ontology aspect.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument('-r', '--resume', required=True, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-d", "--device", type=str, default='cpu',
                        help="Specify what device you'd like to use e.g. 'cpu', 'gpu0' etc.")
    parser.add_argument('-l', "--limit", type=int, default=None,
                        help="How many sequences for evaluation.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']

    asp = args.aspect
    onto = init_GO(asp)
    set_ontology(onto)  # set for seq2go
    lim = args.limit
    ckptpath = args.out_dir

    set_use_cuda('gpu' in args.device)
    set_show_attn(False)

    with open(os.path.join(ckptpath, "kmer-lang-%s.pkl" % GoAspect(asp)), 'rb') as f:
        input_lang = pickle.load(f)
        set_input_lang(input_lang)

    with open(os.path.join(ckptpath, "go-lang-%s.pkl" % GoAspect(asp)), 'rb') as f:
        output_lang = pickle.load(f)
        set_output_lang(output_lang)

    encoder, decoder = init_encoder_decoder(input_lang.n_words, output_lang.n_words)

    load_encoder_decoder_weights(encoder, decoder, args.resume)

    _, _, valid_sequences, valid_annotations = load_training_and_validation(db)
    gen = KmerGoPairsGen(KMER, valid_sequences, valid_annotations, emb=None)

    predictions = {}
    n = len(valid_sequences)
    for i, (seqid, target, _) in enumerate(gen):
        predictions[seqid] = predict(encoder, decoder, target)
        sys.stdout.write("\r{0:.0f}%".format(100.0 * i / n))

    np.save("pred-seq2go-%s.npy" % GoAspect(asp), predictions)

