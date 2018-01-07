import os
import sys

from tqdm import tqdm

import numpy as np

from pymongo import MongoClient

from .seq2go_train import *
from .baselines import *
from .baselines import predict as bl


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


def combine_probabilities(go2probs):
    for go, ps in go2probs.items():
        go2probs[go] = 1 - np.prod([(1 - p) for p in ps])


def predict(encoder, decoder, seq, max_length=MAX_LENGTH):
    input_lengths = [len(seq)]
    seqix = [indexes_from_sequence(input_lang, seq)]
    batch = Variable(torch.LongTensor(seqix), volatile=True).transpose(0, 1)

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(batch, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    # Store output words and attention states
    decoded_words = []
    attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        attentions[di, :attn.size(2)] += attn.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        _, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, attentions[:di + 1, :len(encoder_outputs)]


def predict_proba(encoder, decoder, seq, max_length=MAX_LENGTH, eps=1e-4):

    input_lengths = [len(seq)]
    seqix = [indexes_from_sequence(input_lang, seq)]
    batch = Variable(torch.LongTensor(seqix), volatile=True).transpose(0, 1)

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(batch, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    all_decoder_outputs = []

    # Run through decoder one time step at a time
    for t in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # Choose top word from output
        _, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        if ni == EOS_token:
            break
        else:
            all_decoder_outputs.append(softmax(decoder_output).data[0])

        decoder_input = Variable(torch.LongTensor([ni]))    # Next input is chosen word

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    probs = {}
    for t in range(len(all_decoder_outputs)):
        for ni, pr in enumerate(all_decoder_outputs[t]):
            if pr < eps or ni == EOS_token or ni == SOS_token or ni == PAD_token:
                continue
            leaf = output_lang.index2word[ni]
            for go in onto.propagate([leaf], include_root=False):
                if go in probs:
                    probs[go].append(pr)
                else:
                    probs[go] = [pr]

    combine_probabilities(probs)
    return probs


def softmax(logits):
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # probs_flat: (batch * max_len, 1)
    probs_flat = F.softmax(logits_flat)
    return probs_flat


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("-a", "--aspect", type=str, choices=['F', 'P', 'C'],
                        required=True, help="Specify the ontology aspect.")
    parser.add_argument("-o", "--out_dir", type=str, required=False,
                        default=gettempdir(), help="Specify the output directory.")
    parser.add_argument('-r', '--resume', required=True, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Limit the size of benchmark.")
    parser.add_argument('--predict_proba', action='store_true', default=False,
                        help="Predict probabilities?")
    parser.add_argument("--fallback", type=str, choices=['naive', 'blast'],
                        default="naive", help="Specify a fallback baseline method.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']

    asp = args.aspect
    onto = init_GO(asp)
    set_ontology(onto)  # set for seq2go
    ckptpath = args.out_dir
    lim = args.limit

    set_use_cuda(False)
    set_show_attn(False)

    with open(os.path.join(ckptpath, "kmer-lang-%s.pkl" % GoAspect(asp)), 'rb') as f:
        input_lang = pickle.load(f)
        set_input_lang(input_lang)

    with open(os.path.join(ckptpath, "go-lang-%s.pkl" % GoAspect(asp)), 'rb') as f:
        output_lang = pickle.load(f)
        set_output_lang(output_lang)

    train_seqs, train_annots, valid_sequences, valid_annotations = \
        load_training_and_validation(db, limit=lim)
    gen = KmerGoPairsGen(KMER, valid_sequences, valid_annotations, emb=None)

    encoder, decoder = init_encoder_decoder(input_lang.n_words, output_lang.n_words)

    load_encoder_decoder_weights(encoder, decoder, args.resume)

    print("Predicting...")
    targets = list(gen)
    pbar = tqdm(range(len(targets)), desc="targets processed")

    predictions = {}
    for i, (seqid, inp, out) in enumerate(targets):
        pbar.update(1)
        blen = (MIN_LENGTH <= len(inp) <= MAX_LENGTH) and (MIN_LENGTH <= len(out) <= MAX_LENGTH)
        binp = np.any([word not in input_lang.word2index for word in inp])
        bout = np.any([word not in output_lang.word2index for word in out])
        if binp or bout or not blen:
            tgt = {seqid: valid_sequences[seqid]}
            bl_preds = bl(train_seqs, train_annots, tgt, args.fallback, load_file=False)
            predictions[seqid] = {go: [pr] for go, pr in bl_preds.items()}
            continue
        if args.predict_proba:
            preds = predict_proba(encoder, decoder, inp)
            if seqid in predictions:
                for go, prob in preds.items():
                    if go in predictions[seqid]:
                        predictions[seqid][go].append(prob)
                    else:
                        predictions[seqid][go] = [prob]
            else:
                predictions[seqid] = {go: [pr] for go, pr in preds.items()}
        else:
            annots, _ = predict(encoder, decoder, inp)
            annots = onto.propagate(annots, include_root=False)
            if seqid in predictions:
                for go in annots:
                    if go in predictions[seqid]:
                        predictions[seqid][go].append(1/KMER)
                    else:
                        predictions[seqid][go] = [1/KMER]
            else:
                predictions[seqid] = {go: [1/KMER] for go in annots}

    pbar.close()

    for seqid, preds in predictions.items():
        combine_probabilities(predictions[seqid])

    pth = os.path.join(ckptpath, "pred-seq2go-%s.npy" % GoAspect(asp))
    np.save(pth, predictions)
