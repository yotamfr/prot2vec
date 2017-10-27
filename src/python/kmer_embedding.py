import os
import numpy as np
from gensim.models.word2vec import Word2Vec
from pymongo import MongoClient

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--kmer", type=int, required=True,
                    help="Give the length of the kmer.")
parser.add_argument("-c", "--context", type=int, required=True,
                    help="Give the length of the context window.")
parser.add_argument("-d", "--emb_dim", type=int, required=True,
                    help="Give the dimension of the embedding vector.")
parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                    help="Supply the URL of MongoDB")
parser.add_argument("-s", "--source", type=str, choices=['uniprot', 'sprot'],
                    default="sprot", help="Give source name.")
parser.add_argument("-o", "--outputdir", type=str,  required=False,
                    default="models", help="Specify the output directory")
parser.add_argument("-t", "--num_threads", type=int,  required=False,
                    default=4, help="Specify the output directory")
args = parser.parse_args()

client = MongoClient(args.mongo_url)
db_name = 'prot2vec'
collection = client[db_name][args.source]

ckptpath = args.outputdir
if not os.path.exists(ckptpath):
    os.makedirs(ckptpath)


class KmerSentences(object):

    def __init__(self, kmer):
        self.k = kmer

    @staticmethod
    def get_ngram_sentences(seq, n, offset=0):
        return (
            list(filter(
                lambda ngram: len(ngram) == n, [seq[i:min(i + n, len(seq))]
                                                for i in range(offset, len(seq), n)]
            ))
        )

    def __iter__(self):
        for seq in map(lambda p: p['sequence'], collection.find({})):
            for o in range(self.k):
                yield KmerSentences.get_ngram_sentences(seq, self.k, o)


class Word2VecWrapper(object):

    def __init__(self, ngram_size, win_size, dim_size, min_count=2):

        s = args.source
        t = args.num_threads
        k, c, d, mc = ngram_size, win_size, dim_size, min_count

        unique_str = "%s_%s-mer_dim=%s_c=%s_mc=%s" % (s, k, d, c, mc)
        model_filename = "%s/%s.emb" % (ckptpath, unique_str)
        if os.path.exists(model_filename):
            self._model = Word2Vec.load(model_filename)
        else:
            self._model = Word2Vec(KmerSentences(k),
                                   size=d,
                                   window=c,
                                   min_count=mc,
                                   workers=t, sg=1)
            self._model.save(model_filename)

    def similarity(self, w1, w2):
        return self._model.similarity(w1, w2)

    def __getitem__(self, key):
        return np.array(self._model[key], dtype=np.float64)

    def __contains__(self, key):
        return key in self._model

    @property
    def model(self):
        return self._model


if __name__=="__main__":
    Word2VecWrapper(args.kmer, args.context, args.emb_dim)