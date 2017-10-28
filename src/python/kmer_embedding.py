import os
import operator
import numpy as np
from itertools import combinations
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
from pymongo import MongoClient
import argparse

np.random.seed(1809)

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--kmer", type=int, required=True,
                    help="Give the length of the kmer.")
parser.add_argument("-w", "--win_size", type=int, required=True,
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
parser.add_argument('--train', action='store_true', default=False,
                    help="Specify whether to retrain the model.")
parser.add_argument('--stats', action='store_true', default=False,
                    help="Print statistics when done training.")
args = parser.parse_args()

client = MongoClient(args.mongo_url)
db_name = 'prot2vec'
collection = client[db_name][args.source]

ckptpath = args.outputdir
if not os.path.exists(ckptpath):
    os.makedirs(ckptpath)


AA = [u'A', u'C', u'E', u'D', u'G', u'F', u'I', u'H', u'K', u'M', u'L',
      u'N', u'Q', u'P', u'S', u'R', u'T', u'W', u'V', u'Y', u'X']


class KmerSentences(object):

    def __init__(self, kmer):
        self.k = kmer

    @staticmethod
    def get_ngram_sentences(seq, n, offset=0):
        return list(filter(
            lambda ngram: len(ngram) == n, [seq[i:min(i + n, len(seq))]
                                            for i in range(offset, len(seq), n)]
        ))

    def __iter__(self):
        sample = collection.find({})
        for doc in sample:
            for o in range(self.k):
                seq = doc["sequence"]
                sent = KmerSentences.get_ngram_sentences(seq, self.k, o)
                yield sent

    @staticmethod
    def get_file_stream(k):
        src = args.source
        fname = "%s/%s_%s-mer.sentences" % (ckptpath, src, k)
        sequences = map(lambda doc: doc["sequence"], collection.find({}))
        sentences = (" ".join(KmerSentences.get_ngram_sentences(seq, k, o))
                         for seq in sequences for o in range(k))
        mode = 'r' if os.path.exists(fname) else 'w+'
        with open(fname, mode) as f:
            if mode == 'w+':
                f.writelines(sentences)    # file was created
            stream = LineSentence(f)
            return stream


class Word2VecWrapper(object):

    def __init__(self, ngram_size, win_size, dim_size, min_count=2):

        s = args.source
        t = args.num_threads
        k, c, d, mc = ngram_size, win_size, dim_size, min_count

        unique_str = "%s_%s-mer_dim%s_win%s_mc%s" % (s, k, d, c, mc)
        model_filename = "%s/%s.emb" % (ckptpath, unique_str)
        if not args.train and os.path.exists(model_filename):
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
    def vocab(self):
        return self._model.wv.vocab

    def kmeans(self, k):
        w2v = self
        keys = np.array(list(w2v.vocab.keys()))
        vectors = np.array([w2v[aa] for aa in keys])
        km = KMeans(n_clusters=k).fit(vectors)
        return keys, km.labels_

    def stats(self, k):
        keys, labels = self.kmeans(k)
        clstr = '\n'.join("cluster %s: %s" %
                         (lbl, ' '.join(keys[labels == lbl]))
                         for lbl in np.unique(labels))
        cs = combinations(keys, 2)
        ds = {c: self.similarity(c[0], c[1]) for c in cs}
        hi_i = max(ds.items(), key=operator.itemgetter(1))[0]
        lo_i = min(ds.items(), key=operator.itemgetter(1))[0]
        av = np.mean(list(ds.values()))
        hi_s = "highest similarity: sim(%s, %s)=%s" % (hi_i[0], hi_i[1], ds[hi_i])
        lo_s = "lowest similarity: sim(%s, %s)=%s" % (lo_i[0], lo_i[1], ds[lo_i])
        av_s = "average similarity: %s" % av
        return '\n'.join([clstr, hi_s, lo_s, av_s])


if __name__=="__main__":
    print("Training W2V...")
    w2v = Word2VecWrapper(args.kmer, args.win_size, args.emb_dim)
    print("Done Training!")
    if args.stats:
        print(w2v.stats(8))
    print("Finished!")
