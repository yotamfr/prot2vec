from src.python.seq2go_train import *


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
        epoch = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        return encoder, decoder, epoch
    else:
        print("=> no checkpoint found at '%s'" % args.resume)


# def predict(asp):
#     init_GO(asp)
#     lim = None
#     seqs_train, annots_train, seqs_valid, annots_valid = \
#         load_training_and_validation(db, lim)
#     perf = {}
#     for meth in methods:
#         pred = predict(seqs_train, annots_train, seqs_valid, meth)
#         perf[meth] = performance(pred, annots_valid)
#     plot_precision_recall(perf)


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("-a", "--aspect", type=str, choices=['F', 'P', 'C'],
                        default="F", help="Specify the ontology aspect.")
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

    asp = args.aspect
    onto = init_GO(asp)
    set_ontology(onto)  # set for seq2go
    lim = args.limit
    ckptpath = args.out_dir

    set_use_cuda('gpu' in args.device)
    set_show_attn(False)

    # input_lang = Lang("KMER")
    # output_lang = Lang("GO")
    # set_input_lang(input_lang)
    # set_output_lang(output_lang)

    # seqid2seq, _, seqid2goid = load_data(db, asp, limit=None)
    # lang_gen = KmerGoPairsGen(KMER, seqid2seq, seqid2goid, emb=None)
    # prepare_data(lang_gen)
    #
    # input_lang.trim(MIN_COUNT)
    # output_lang.trim(MIN_COUNT)

    with open(os.path.join(ckptpath, "kmer-lang-%s.pkl" % GoAspect(asp)), 'rb') as f:
        input_lang = pickle.load(f)
        set_input_lang(input_lang)

    with open(os.path.join(ckptpath, "go-lang-%s.pkl" % GoAspect(asp)), 'rb') as f:
        output_lang = pickle.load(f)
        set_output_lang(output_lang)

    encoder, decoder = init_encoder_decoder(input_lang.n_words, output_lang.n_words)

    encoder, decoder, _ = load_encoder_decoder_weights(encoder, decoder, args.resume)
