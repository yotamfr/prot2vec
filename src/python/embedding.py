import os
import sys
import operator
import numpy as np
from itertools import combinations

from gensim.models.word2vec import Word2Vec
from gensim import matutils

import tensorflow as tf

from pymongo.errors import CursorNotFound
from pymongo import MongoClient
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import matplotlib.pyplot as plt
except ImportError as err:
    plt = None
    print(err)

from tempfile import gettempdir

import argparse

np.random.seed(1809)

AA = 'E N L I R Y K Q T F D M S U X B O V A W G P Z H C'.split()
reverse_dictionary = dict(zip(range(len(AA)), AA))
dictionary = dict(zip(AA, range(len(AA))))
vocabulary_size = 25
n_clstr = 8

valid_examples = [
    dictionary['A'],
    dictionary['V'],
    dictionary['K'],
    dictionary['R'],
    dictionary['H'],
    dictionary['C']
]


class SequenceBatchLoader(object):
    
    def __init__(self, win_size, batch_size):
        self.win_size = win_size
        self.batch_size = batch_size
        self.batch_buffer = np.ndarray(shape=(batch_size,), dtype=np.int32)
        self.labels_buffer = np.ndarray(shape=(batch_size,), dtype=np.int32)

    def __iter__(self):
        
        self.stream = map(lambda p: p['sequence'], collection.find({}))

        seq = self._get_sequence()
        seq_pos = 0

        batch_buffer, labels_buffer = self.batch_buffer, self.labels_buffer

        while len(seq):

            batch_buffer, labels_buffer, seq_pos, batch_pos = \
                self._get_batch(seq, batch_buffer, labels_buffer, batch_pos=0, seq_pos=seq_pos)

            if seq_pos == 0:   # seq finished
                try:
                    seq = self._get_sequence()
                except CursorNotFound:
                    break

                batch_buffer, labels_buffer, seq_pos, batch_pos = \
                    self._get_batch(seq, batch_buffer, labels_buffer, batch_pos=batch_pos, seq_pos=0)

            else:
                yield batch_buffer, labels_buffer
            
    def _get_batch(self, seq, batch, labels, batch_pos=0, seq_pos=0):
        return batch, labels, seq_pos, batch_pos
    
    def _get_sequence(self):
        return ""
    

class CbowBatchLoader(SequenceBatchLoader):

    def __init__(self, win_size, batch_size):
        super(CbowBatchLoader, self).__init__(win_size, batch_size)
        self.mask = np.array(range(2*win_size+1)) != win_size
        self.batch_buffer = np.ndarray(shape=(batch_size, win_size * 2), dtype=np.int32)

    def _get_sequence(self):
        return np.array(list(map(lambda aa: dictionary[aa], next(self.stream))))
        
    def _get_batch(self, seq, batch, labels, batch_pos=0, seq_pos=0):
        win = self.win_size
        mask = self.mask
        if seq_pos == 0:
            seq_pos = win
        batch_size = self.batch_size
        while batch_pos < batch_size:
            if seq_pos + win >= len(seq):    # seq finished before batch
                return batch, labels, 0, batch_pos
            if batch_pos == batch_size:     # batch finished before seq
                break
            start = seq_pos-win
            end = seq_pos + win + 1
            context = seq[start:end]
            batch[batch_pos, :] = context[mask]
            labels[batch_pos] = context[win]
            batch_pos += 1
            seq_pos += 1

        return batch, labels, seq_pos, batch_pos


class SkipGramBatchLoader(SequenceBatchLoader):

    def __init__(self, win_size, batch_size):
        super(SkipGramBatchLoader, self).__init__(win_size, batch_size)

    def _get_sequence(self):
        return next(self.stream, None)

    def _get_batch(self, seq, batch, labels, batch_pos=0, seq_pos=0):
        batch_size = self.batch_size
        possible_context_sizes = range(1, self.win_size + 1)
        while batch_pos < batch_size:
            win = np.random.choice(possible_context_sizes)
            for offset in range(-win, win+1):
                label_pos = seq_pos + offset
                if offset == 0 or label_pos < 0:
                    continue
                if label_pos >= len(seq):
                    continue
                if batch_pos == batch_size:     # batch finished before seq
                    break
                if seq_pos == len(seq):     # seq finished before batch
                    return batch, labels, 0, batch_pos
                labels[batch_pos] = dictionary[seq[label_pos]]
                batch[batch_pos] = dictionary[seq[seq_pos]]
                batch_pos += 1
            seq_pos += 1

        return batch, labels, seq_pos, batch_pos


class KmerSentencesLoader(object):

    def __init__(self, kmer):
        self.k = kmer

    @staticmethod
    def get_ngram_sentences(seq, n, offset=0):
        return list(filter(
            lambda ngram: len(ngram) == n, [seq[i:min(i + n, len(seq))]
                                            for i in range(offset, len(seq), n)]
        ))

    def __iter__(self):
        n = collection.count({})
        pbar = tqdm(range(n), desc="sequences loaded")
        stream = map(lambda p: p['sequence'], collection.find({}))
        for seq in stream:
            for o in range(self.k):
                yield KmerSentencesLoader.get_ngram_sentences(seq, self.k, o)
            pbar.update(1)
        pbar.close()


class Word2VecWrapper(object):

    def __init__(self, model, kmer_size, win_size, dim_size, min_count=2,
                 src='sprot', n_threads=3, b_train=False, ckptpath='models'):

        t = n_threads
        k, c, d, mc = kmer_size, win_size, dim_size, min_count

        unique_str = "gensim_%s_%s_%s-mer_dim%s_win%s" % (model, src, k, d, c)
        model_filename = "%s/%s.emb" % (ckptpath, unique_str)
        if not b_train and os.path.exists(model_filename):
            self._model = Word2Vec.load(model_filename)
        else:
            print("Training %s on %s (size=%s, window=%s, min_count=%s, workers=%s)"
                  % (model, src, d, c, mc, t))
            self._model = Word2Vec(KmerSentencesLoader(k),
                                   size=d,
                                   window=c,
                                   min_count=mc,
                                   workers=t,
                                   sg=(model == 'skipgram'))
            self._model.save(model_filename)

    def similarity(self, w1, w2):
        return self._model.similarity(w1, w2)

    def __getitem__(self, key):
        return np.array(self._model[key], dtype=np.float64)

    def __contains__(self, key):
        return key in self._model

    @property
    def embeddings(self):
        return np.array([self[reverse_dictionary[i]] for i in range(vocabulary_size)])

    @property
    def vocab(self):
        return self._model.wv.vocab


class Embedder(object):

    def __init__(self,
                 model_name,
                 loader,
                 win_size,  # How many words to consider left and right.
                 emb_size,  # Dimension of the embedding vector.
                 batch_size=32,
                 hidden_size=512):

        self.emb_size = emb_size
        self.win_size = win_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.name = model_name

        emb = tf.get_variable("emb",
                              [vocabulary_size, emb_size],
                              initializer=tf.zeros_initializer)

        if os.path.exists(os.path.join(gettempdir(), '%s.ckpt.meta' % self.name)) and not args.train:
            with tf.Session() as sess:
                    saver = tf.train.Saver()
                    saver.restore(sess, os.path.join(gettempdir(), '%s.ckpt' % self.name))
                    self.embeddings = emb.eval()
        else:
            self._train(loader)

    def _train(self, loader):

        tf_graph, train_inputs, train_labels, embeddings, loss_fn = self._init_tf_graph()

        with tf_graph.as_default():

            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       100000, 0.96, staircase=True)
            optimizer = (tf.train.GradientDescentOptimizer(learning_rate)
                         .minimize(loss_fn, global_step=global_step))

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = tf.divide(embeddings, norm)
            similarity = tf.matmul(normalized_embeddings, tf.transpose(normalized_embeddings))

            initializer = tf.global_variables_initializer()

        with tf.Session(graph=tf_graph) as sess:
            initializer.run()
            average_loss = 0
            optimal_loss = np.inf

            for step, (batch_inputs, batch_labels) in enumerate(loader):

                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                _, loss_val = sess.run([optimizer, loss_fn], feed_dict=feed_dict)
                average_loss += loss_val

                mod = 1000
                if step > 0 and step % mod == 0:

                    self.embeddings = normalized_embeddings.eval()
                    self.sim = similarity.eval()

                    average_loss /= mod
                    average_similarity = np.mean(self.sim)
                    msg = 'Step : %d Average Loss : %6.4f Average Similarity : %.4f' \
                          % (step,  average_loss, average_similarity)

                    if args.verbose:
                        print(msg)
                        sim = similarity.eval()
                        for i in range(len(valid_examples)):
                            valid_word = reverse_dictionary[valid_examples[i]]
                            top_k = 3  # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = reverse_dictionary[nearest[k]]
                                log_str = '%s %s,' % (log_str, close_word)
                            print(log_str)

                    else:
                        sys.stdout.write("\r" + msg)

                    if average_loss < optimal_loss:
                        optimal_loss = average_loss
                        self.save(sess)

                    average_loss = 0

            self.embeddings = normalized_embeddings.eval()
            self.sim = similarity.eval()
            self.save(sess)

    def save(self, sess):
        saver = tf.train.Saver()
        save_path = os.path.join(gettempdir(), '%s.ckpt' % self.name)
        if args.verbose: print("Saving to %s" % save_path)
        return saver.save(sess, save_path)

    def _init_tf_graph(self):
        return tf.Graph(), None, None, None, None

    def similarity(self, aa1, aa2):
        return np.dot(matutils.unitvec(self[aa1]), matutils.unitvec(self[aa2]))
        # return self.sim[dictionary[aa1], dictionary[aa2]]

    def __getitem__(self, aa):
        return self.embeddings[dictionary[aa]]

    def __contains__(self, aa):
        return aa in dictionary

    @property
    def vocab(self):
        keys = list(dictionary.keys())
        values = [self[aa] for aa in keys]
        return dict(zip(keys, values))


class CBOW(Embedder):

    def __init__(self,
                 win_size,  # How many words to consider left and right.
                 emb_size,  # Dimension of the embedding vector.
                 batch_size=8):

        my_name = "my_cbow_%s_1-mer_dim%s_win%s" % (args.source, emb_size, win_size)
        super(CBOW, self).__init__(my_name, CbowBatchLoader(win_size, batch_size),
                                   win_size, emb_size, batch_size)

    def _init_tf_graph(self):

        graph = tf.Graph()
        emb_size = self.emb_size
        win_size = self.win_size
        batch_size = self.batch_size
        hidden_size = self.hidden_size

        with graph.as_default():
            train_inp = tf.placeholder(tf.int32, shape=[batch_size, win_size * 2])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size])

            proj = tf.get_variable("emb", initializer=tf.random_uniform([vocabulary_size, emb_size], -1.0, 1.0))

            # x = tf.reduce_mean([tf.nn.embedding_lookup(proj, train_inp[:, j])
            #                     for j in range(win_size * 2)], axis=0)
            x = tf.concat([tf.nn.embedding_lookup(proj, train_inp[:, j])
                           for j in range(win_size * 2)], axis=1)

            w1 = tf.Variable(tf.random_uniform([emb_size * win_size * 2, hidden_size], -1.0, 1.0))
            b1 = tf.Variable(tf.zeros([hidden_size]))

            o1 = tf.nn.bias_add(tf.matmul(x, w1), b1)
            o1 = tf.contrib.layers.batch_norm(o1)
            h1 = tf.nn.sigmoid(o1)

            w2 = tf.Variable(tf.random_uniform([hidden_size, hidden_size // 2], -1.0, 1.0))
            b2 = tf.Variable(tf.zeros([hidden_size // 2]))

            o2 = tf.nn.bias_add(tf.matmul(h1, w2), b2)
            o2 = tf.contrib.layers.batch_norm(o2)
            h2 = tf.nn.sigmoid(o2)

            w3 = tf.Variable(tf.random_uniform([hidden_size // 2, hidden_size], -1.0, 1.0))
            b3 = tf.Variable(tf.zeros([hidden_size]))

            o3 = tf.nn.bias_add(tf.matmul(h2, w3), b3)
            o3 = tf.contrib.layers.batch_norm(o3)
            h3 = tf.nn.sigmoid(o3)

            w4 = tf.Variable(tf.random_uniform([hidden_size, vocabulary_size], -1.0, 1.0))
            b4 = tf.Variable(tf.zeros([vocabulary_size]))

            y_hat = tf.nn.bias_add(tf.matmul(h3, w4), b4)

            y = tf.one_hot(train_labels, vocabulary_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
            loss_fn = tf.reduce_sum(loss, axis=0)

        return graph, train_inp, train_labels, proj, loss_fn


class SkipGram(Embedder):

    def __init__(self,
                 win_size,  # How many words to consider left and right.
                 emb_size=20,  # Dimension of the embedding vector.
                 batch_size=8):
        my_name = "my_sg_%s_1-mer_dim%s_win%s" % (args.source, emb_size, win_size)
        super(SkipGram, self).__init__(my_name, SkipGramBatchLoader(win_size, batch_size),
                                       win_size, emb_size, batch_size)

    def _init_tf_graph(self):

        graph = tf.Graph()
        emb_size = self.emb_size
        batch_size = self.batch_size
        hidden_size = self.hidden_size

        with graph.as_default():
            train_inp = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size])

            proj = tf.Variable(tf.random_uniform([vocabulary_size, emb_size], -1.0, 1.0))

            x = tf.nn.embedding_lookup(proj, train_inp)

            w1 = tf.Variable(tf.random_uniform([emb_size, hidden_size], -1.0, 1.0))
            b1 = tf.Variable(tf.zeros([hidden_size]))

            o1 = tf.nn.bias_add(tf.matmul(x, w1), b1)
            o1 = tf.contrib.layers.batch_norm(o1)
            h1 = tf.nn.leaky_relu(o1)

            w2 = tf.Variable(tf.random_uniform([hidden_size, hidden_size], -1.0, 1.0))
            b2 = tf.Variable(tf.zeros([hidden_size]))

            o2 = tf.nn.bias_add(tf.matmul(h1, w2), b2)
            o2 = tf.contrib.layers.batch_norm(o2)
            h2 = tf.nn.sigmoid(o2)

            w3 = tf.Variable(tf.random_uniform([hidden_size, vocabulary_size], -1.0, 1.0))
            b3 = tf.Variable(tf.zeros([vocabulary_size]))

            y_hat = tf.nn.bias_add(tf.matmul(h2, w3), b3)

            y = tf.one_hot(train_labels, vocabulary_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
            loss_fn = tf.reduce_sum(loss, axis=0)

        return graph, train_inp, train_labels, proj, loss_fn


def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
    plt.show()


def pca(embeddings):
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)


def tsne(embeddings):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    return tsne.fit_transform(embeddings[:vocabulary_size, :])


def plot(low_dim_embs):
    labels = [reverse_dictionary[i] for i in range(vocabulary_size)]
    path = os.path.join(gettempdir(), 'plot.png')
    plot_with_labels(low_dim_embs, labels, path)
    print("Saving to %s" % path)


def kmeans(w2v, k):
    keys = np.array(list(w2v.vocab.keys()))
    vectors = np.array([w2v[aa] for aa in keys])
    km = KMeans(n_clusters=k).fit(vectors)
    return keys, km.labels_


def clstr_stats(w2v, k):
    keys, labels = kmeans(w2v, k)
    clstr = '\n'.join("cluster %s: %s" %
                      (lbl, ' '.join(keys[labels == lbl]))
                      for lbl in np.unique(labels))
    cs = combinations(keys, 2)
    ds = {c: w2v.similarity(c[0], c[1]) for c in cs}
    hi_i = max(ds.items(), key=operator.itemgetter(1))[0]
    lo_i = min(ds.items(), key=operator.itemgetter(1))[0]
    av = np.mean(list(ds.values()))
    hi_s = "highest similarity: sim(%s, %s)=%s" % (hi_i[0], hi_i[1], ds[hi_i])
    lo_s = "lowest similarity: sim(%s, %s)=%s" % (lo_i[0], lo_i[1], ds[lo_i])
    av_s = "average similarity: %s" % av
    return '\n'.join([clstr, hi_s, lo_s, av_s])


def add_arguments(parser):
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
    parser.add_argument("-m", "--model", type=str, choices=['cbow', 'skipgram', 'gensim'],
                        default="gensim", help="Choose what type of model to use.")
    parser.add_argument("-o", "--outputdir", type=str, required=False,
                        default="models", help="Specify the output directory")
    parser.add_argument("-t", "--num_threads", type=int, required=False,
                        default=4, help="Specify the output directory")
    parser.add_argument('--train', action='store_true', default=False,
                        help="Specify whether to retrain the model.")
    parser.add_argument('--stats', action='store_true', default=False,
                        help="Print statistics when done training.")
    parser.add_argument("-v", '--verbose', action='store_true', default=False,
                        help="Run in verbose mode.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    collection = client['prot2vec'][args.source]

    ckptpath = args.outputdir
    if not os.path.exists(ckptpath):
        os.makedirs(ckptpath)

    if args.model == 'cbow':
        w2v = CBOW(args.win_size, args.emb_dim, batch_size=8)

    elif args.model == 'skipgram':

        w2v = SkipGram(args.win_size, args.emb_dim, batch_size=8)

    elif args.model == 'gensim':

        w2v = Word2VecWrapper(args.model, args.kmer,
                              args.win_size, args.emb_dim,
                              n_threads=args.num_threads,
                              b_train=args.train,
                              ckptpath=ckptpath,
                              src=args.source)

    else:
        print("Unknown model")
        exit(1)

    if args.stats and plt:
        plot(tsne(w2v.embeddings))
        plot(pca(w2v.embeddings))
    if args.stats:
        print(clstr_stats(w2v, n_clstr))
