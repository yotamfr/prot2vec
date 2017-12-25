import os
import sys

import numpy as np

from gensim.models.word2vec import Word2Vec

from tqdm import tqdm

from tempfile import gettempdir
DATA_ROOT = gettempdir()


def get_kmer_sentences(seq, k, offset=0):
    return list(filter(
        lambda kmer: len(kmer) == k, [seq[i:min(i + k, len(seq))]
                                      for i in range(offset, len(seq), k)]))


class KmerSentencesLoader(object):

    def __init__(self, kmer, corpus):
        self.k = kmer
        self.corpus = corpus

    def __iter__(self):
        corpus = self.corpus
        n = len(corpus)
        pbar = tqdm(range(n), desc="sequences loaded")
        for seq in corpus:
            for o in range(self.k):
                yield get_kmer_sentences(seq, self.k, o)
            pbar.update(1)
        pbar.close()


class Kmer2Vec(object):

    def __init__(self, db, kmer_size, win_size=25, vec_size=100, min_count=2,
                 src='sp', arch='sg', n_threads=3, b_train=False):

        t = n_threads
        k, c, d, mc = kmer_size, win_size, vec_size, min_count

        unique_str = "gensim-%s-%s-%smer-dim%s-win%s" % (arch, src, k, d, c)
        model_filename = "%s/%s.emb" % (DATA_ROOT, unique_str)
        if not b_train and os.path.exists(model_filename):
            self._model = Word2Vec.load(model_filename)
        else:
            print("Training %s on %s (size=%s, window=%s, min_count=%s, workers=%s)"
                  % (arch, src, d, c, mc, t))

            stream = map(lambda p: p['sequence'], db.uniprot.find({'db': src}))

            self._model = Word2Vec(KmerSentencesLoader(k, list(stream)),
                                   size=d,
                                   window=c,
                                   min_count=mc,
                                   workers=t,
                                   sg=(arch == 'sg'))
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
