#!/usr/bin/env python

from gensim import matutils
from gensim.models import doc2vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
from pymongo import MongoClient
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

from models import EcodDomain
from models import Uniprot
from models import PdbChain

assert doc2vec.FAST_VERSION > -1

from numpy import dot

from sklearn.decomposition import PCA

import itertools
import networkx as nx
import pandas as pd
import numpy as np
import json
import os

import utils
import parameters as params

args = params.arguments
logger = utils.get_logger("prot2vec")

ckptpath = args["ckpt_path"]
seq_length = args["seq_length"]
emb_dim = args["word_embedding_dim"]
datapath = args["data_path"]

ECOD = args['ecod_fasta']
PDB = args['pdb_fasta']
PDBNR = args['pdbnr_fasta']
UNIPROT = args['uniprot_sprot_fasta']

client = MongoClient('mongodb://localhost:27017/')
dbname = args["db"]
db = client[dbname]


consts = {
    "EDGE_GRAPHICS": lambda: {
        'width': 1.0,
        'fill': '"#889999"',
        'type': '"line"'
    },
    "NODE_GRAPHICS": lambda: {
        'x': np.random.randint(-10e3, 10e3)
        , 'y': np.random.randint(-10e3, 10e3)
        , 'w': 20.0, 'h': 20.0
        , 'type': '"ellipse"', 'fill': '"#0000ff"'
        , 'outline': '"#666666"'
    }
}


def get_cdhit_clusters(ratio, fasta_filename, parse=lambda seq: seq.split('>')[1].split('...')[0].split('|')[0]):

    cluster_filename = "%s.%s" % (fasta_filename, ratio)

    if not os.path.exists(cluster_filename):
        os.system("cdhit -i %s -o %s -c %s -n 5" % (fasta_filename, cluster_filename, ratio))

    # open the cluster file and set the output dictionary
    cluster_file, cluster_dic, reverse_dic = open("%s.clstr" % cluster_filename), {}, {}

    logger.info("Reading cluster groups...")
    # parse through the cluster file and store the cluster name + sequences in the dictionary
    cluster_groups = (x[1] for x in itertools.groupby(cluster_file, key=lambda line: line[0] == '>'))
    for cluster in cluster_groups:
        name = int(next(cluster).strip().split()[-1])
        ids = [parse(seq) for seq in next(cluster_groups)]
        cluster_dic[name] = ids
    for cluster, ids in cluster_dic.items():
        for seqid in ids:
            reverse_dic[seqid] = cluster
    with open('%s.%s.json' % (fasta_filename, ratio), 'w') as f:
        json.dump(reverse_dic, f)
    logger.info("Detected %s clusters (>%s similarity) groups..." % (len(cluster_dic), ratio))

    return cluster_dic, reverse_dic


class Graph(object):

    def __init__(self, edgelist):
        self.graph = nx.Graph()
        self.edgelist = edgelist
        if edgelist and os.path.exists(edgelist):
            self.from_edgelist(edgelist)
        else:
            self.init()

    def init(self):
        pass

    def is_connected(self):
        return nx.is_connected(self.graph)

    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

    def add_node(self, node):
        self.graph.add_node(node)

    def nodes(self):
        return self.graph.nodes()

    def edges(self):
        return self.graph.edges()

    def to_edgelist(self, filename):
        nx.write_edgelist(self.graph, filename, data=False)

    def from_edgelist(self, filename):
        self.graph = nx.read_edgelist(filename, data=False)

    def number_connected_components(self):
        return nx.number_connected_components(self.graph)

    def __str__(self):
        return 'G=(V,E) |E|=%s |V|=%s #CCs=%s' %\
               (len(self.edges()), len(self.nodes()), self.number_connected_components())

    def to_gml(self, relpath):
        nx.write_gml(self, relpath)


class RandomGraph(Graph):

    def __init__(self, edgelist_filename, collection, Model):
        self.Model = Model
        self.collection = collection
        super(RandomGraph, self).__init__(edgelist_filename)

    def init(self):
        logger.info(self)
        logger.info("Initiating Graph from collection")
        nodes = list(map(self.Model, self.collection.find({})))
        for i in tqdm(range(len(nodes)), desc="Nodes Processed"):
            self.add_node(nodes[i].name)
        logger.info("Finished!")
        logger.info(self)

    def coalesce(self, func, th=.90, num_trials=2000):
        bar = tqdm(range(num_trials), desc=str(self))
        for i in bar:
            nodes = np.random.choice(self.nodes(), size=1000, replace=False)
            for u, v in itertools.combinations(nodes, 2):
                if hash(u) % 2000 == hash(v) % 2000:
                    func(self, u, v)
            if not i % 50:
                self.to_edgelist(self.edgelist)
                bar.set_description(str(self))


class UniprotGraph(Graph):

    def __init__(self, edgelist_filename):
        super(UniprotGraph, self).__init__(edgelist_filename)

    def init(self):
        logger.info(self)
        logger.info("Initiating Graph from Uniprot collection")
        nodes = list(map(Uniprot, db.uniprot.find({})))
        for i in tqdm(range(len(nodes)), desc="Nodes Processed"):
            self.add_node(nodes[i].uid)
        logger.info("Finished!")
        logger.info(self)

    def coalesce(self, func, th=.90, num_trials=2000):
        parse_uniprot_id = lambda seq: seq.split('>')[1].split('...')[0].split('|')[1]
        cluster_dic, reverse_dic = get_cdhit_clusters(th, UNIPROT, parse_uniprot_id)
        bar = tqdm(range(num_trials), desc=str(self))
        for i in bar:
            nodes = np.random.choice(self.nodes(), size=1000, replace=False)
            for u, v in itertools.combinations(nodes, 2):
                if u in reverse_dic and v in reverse_dic:
                    if reverse_dic[u] == reverse_dic[v]:
                        func(self, u, v)
            if not i % 50:
                self.to_edgelist(self.edgelist)
                bar.set_description(str(self))


class PdbGraph(Graph):

    def __init__(self, edgelist_filename):
        super(PdbGraph, self).__init__(edgelist_filename)

    def init(self):
        logger.info(self)
        logger.info("Initiating Graph from PDB collection")
        nodes = list(map(PdbChain, db.pdb.find({})))
        for i in tqdm(range(len(nodes)), desc="Nodes Processed"):
            self.add_node(nodes[i].name)
            for neighbor in map(PdbChain, db.pdb.find({"complex": nodes[i].complex})):
                self.add_edge(nodes[i].name, neighbor.name)
        logger.info("Finished!")
        logger.info(self)

    def coalesce(self, func, th=.90, num_trials=2000):
        cluster_dic, reverse_dic = get_cdhit_clusters(th, PDB)
        bar = tqdm(range(num_trials), desc=str(self))
        for i in bar:
            nodes = np.random.choice(self.nodes(), size=1000, replace=False)
            for u, v in itertools.combinations(nodes, 2):
                if u in reverse_dic and v in reverse_dic:
                    if reverse_dic[u] == reverse_dic[v]:
                        func(self, u, v)
            if not i % 50:
                self.to_edgelist(self.edgelist)
                bar.set_description(str(self))


class PdbGraphNR(Graph):

    def __init__(self, edgelist_filename):
        super(PdbGraphNR, self).__init__(edgelist_filename)

    def init(self):
        logger.info(self)
        logger.info("Initiating Graph from PDB collection")
        nodes = list(map(PdbChain, db.pdbnr.find({})))
        for i in tqdm(range(len(nodes)), desc="Nodes Processed"):
            self.add_node(nodes[i].name)
            for neighbor in map(PdbChain, db.pdbnr.find({"complex": nodes[i].complex})):
                if nodes[i].name != neighbor.name:
                    self.add_edge(nodes[i].name, neighbor.name)
        logger.info("Finished!")
        logger.info(self)

    def coalesce(self, func, th=.90, num_trials=2000):
        cluster_dic, reverse_dic = get_cdhit_clusters(th, PDBNR)
        bar = tqdm(range(num_trials), desc=str(self))
        for i in bar:
            nodes = np.random.choice(self.nodes(), size=1000, replace=False)
            for u, v in itertools.combinations(nodes, 2):
                if u in reverse_dic and v in reverse_dic:
                    if reverse_dic[u] == reverse_dic[v]:
                        func(self, u, v)
            if not i % 50:
                self.to_edgelist(self.edgelist)
                bar.set_description(str(self))


class EcodGraph(Graph):

    def __init__(self, edgelist_filename):
        super(EcodGraph, self).__init__(edgelist_filename)

    def init(self):
        logger.info(self)
        logger.info("Initiating Graph from ECOD collection")
        nodes = list(map(EcodDomain, db.ecod.find({})))
        for i in tqdm(range(len(nodes)), desc="Nodes Processed"):
            u = nodes[i]
            for v in u.get_adj_nodes():
                self.add_edge(u.name, v.name)
        logger.info("Finished!")
        logger.info(self)

    def coalesce(self, func, th=.90, num_trials=2000):
        cluster_dic, reverse_dic = get_cdhit_clusters(th, ECOD)
        nodes_dic = get_ecod_dictionary("ecod_id")
        bar = tqdm(range(num_trials), desc=str(self))
        for i in bar:
            ids = np.random.choice(self.nodes(), size=1000, replace=False)
            nodes = filter(lambda e: e.uid in reverse_dic,
                           map(lambda e: nodes_dic[e], ids))
            for u, v in itertools.combinations(nodes, 2):
                if reverse_dic[u.uid] == reverse_dic[v.uid]:
                    func(self, u.name, v.name)
            if not i % 50:
                self.to_edgelist(self.edgelist)
                bar.set_description(str(self))


class BioGridGraph(Graph):

    def __init__(self, edgelist_filename, organism_name="Homo_sapiens"):
        self.organism = organism_name
        super(BioGridGraph, self).__init__(edgelist_filename)

    def init(self):
        logger.info(self)
        logger.info("Initiating Graph from BioGrid collection")
        interactions = db.biogrid.find({})
        for interact in interactions:
            u = interact["BioGRID_ID_Interactor_A"]
            v = interact["BioGRID_ID_Interactor_B"]
            self.add_edge(u, v)


# class IntActGraph(Graph):
#
#     def __init__(self, edgelist_filename, organism_name="Homo_sapiens"):
#         self.organism = organism_name
#         super(IntActGraph, self).__init__(edgelist_filename)
#
#     def init(self):
#         logger.info(self)
#         logger.info("Initiating Graph from IntAct collection")
#         interactions = db.intact.find({})
#         for interact in interactions:
#             e = IntActEdge(interact)
#             self.add_edge(e.source, e.target)


def complex(pdb): return pdb[:4]


def chain(pdb): return pdb[4:]


def create_adjacency_list():
    logger.info("Populating adjacency_list")
    adj_list = {}
    nodes = list(db.pdb.find({}))
    for i in tqdm(range(len(nodes)), desc="Sequences Processed"):
        u = nodes[i]
        adj = [complex(v) for v in u["duplicates"]] + [u["complex"]]
        links1 = list(map(lambda v: v["_id"],
                          filter(lambda v: v["_id"] != u["_id"],
                                 db.pdb.find({"complex": {"$in": adj}}))))

        links2 = list(map(lambda v: v["_id"],
                          filter(lambda v: v["_id"] != u["_id"],
                                 [] if "clstr" not in u else db.pdb.find({"clstr": u["clstr"]}))))
        adj_list[u["_id"]] = links1 + links2
    return adj_list


def get_weights(adj_list, vertices):
    degrees = list(map(lambda u: len(adj_list[u]), vertices))
    sum_degs = sum(degrees)
    return list(map(lambda deg: deg / sum_degs, degrees)), sum_degs


def random_walk(adj_list, _src, max_length):
    _path = [_src]
    while len(_path) < max_length:
        _links = None if _src not in adj_list \
            else adj_list[_src]
        if not _links:
            break
        _trg = _links[np.random.randint(0, len(_links))]
        if _trg in _path:  # i.e. hit a circle
            break
        _path.append(_trg)
        _src = _trg
    return _path


def store_path(_path, _epoch):
    db.paths.insert_one({
        "src": _path[0],
        "path": _path,
        "length": len(_path),
        "epoch_data": _epoch,
    })


def power_rule_sample(adj_list, name="power_rule", verbose=False):
    _sources = list(adj_list.keys())
    if len(_sources) == 0:
        return
    weights, sum_degrees = get_weights(adj_list, _sources)
    epochs, _nodes = 1 + int(sum_degrees / len(_sources)), list(adj_list.keys())
    if verbose:
        logger.info("sum(weights)=%s len(weights)=%s" % (sum(weights), len(weights)))
    for e in range(epochs):
        _epoch = {"sample_name": name, "ord": e + 1, "total": epochs}
        logger.info("epoch=%s/%s" % (e + 1, epochs))
        for i in tqdm(range(len(_sources)), desc="Vertices Processed"):
            _src = np.random.choice(_sources, p=weights)
            _path = random_walk(adj_list, _src, max_length=100)
            store_path(_path, _epoch)
            if verbose:
                logger.info("->".join(_path))


def thicken(G, u, v): G.add_edge(u, v)


def dump_sentences():
    num_samples = db.paths.count({})
    filename = "%s/paths.txt" % ckptpath
    sentences = map(lambda d: '%s\n' % ' '.join(d["path"]),
                    db.paths.find({"filename": filename}))
    logger.info("Dumping %s sentences to file: %s" % (num_samples, filename))
    with open(filename, 'w+') as f:
        f.writelines(sentences)
    return filename


def train_word2vec_model():
    corpus_filename = dump_sentences()
    sentences = LineSentence(corpus_filename)
    logger.info("Training Word2Vec...")
    model = Word2Vec(sentences, size=emb_dim, window=10, min_count=5, workers=4, sg=1)
    model_filename = "%s/prot2vec.%s.model" % (ckptpath, emb_dim)
    model.save(model_filename)
    return model, model_filename


def get_word_vectors(model_filename):
    logger.info("extracting vectors...")
    model = Word2Vec.load(model_filename)
    X = np.array([model[word] for word in model.vocab])
    Xt = X.transpose()
    DATA2 = {0: list(model.vocab)}
    for i in range(Xt.shape[0]):
        DATA2[i + 1] = Xt[i]
    pca = PCA(n_components=3)
    pca.fit(X)
    PCAt = pca.transform(X).transpose()
    DATA1 = {0: list(model.vocab)}
    for i in range(PCAt.shape[0]):
        DATA1[i + 1] = PCAt[i]
    df1 = pd.DataFrame(data=DATA1)
    df2 = pd.DataFrame(data=DATA2)

    pca_filename = 'wordvecs_%s_pca_%s.tsv' % (model_filename, emb_dim)
    wordvecs_filename = 'wordvecs_%s_%s.tsv' % (model_filename, emb_dim)
    df1.to_csv("%s/%s" % (ckptpath,pca_filename), sep='\t', header=False, index=False)
    df2.to_csv("%s/%s" % (ckptpath,wordvecs_filename), sep='\t', header=False, index=False)
    return pca_filename, wordvecs_filename


def get_dictionary(collection, Model, field):
    nodes_dic = {}
    nodes = list(collection.find({}))
    for i in tqdm(range(len(nodes)), desc="Docs Processed"):
        nodes_dic[nodes[i][field]] = Model(nodes[i])
    return nodes_dic


def get_ecod_dictionary(key_field):
    logger.info("Computing ECOD Dictionary ...")
    return get_dictionary(db.ecod, EcodDomain, key_field)


def get_pdb_dictionary(key_field):
    logger.info("Computing PDB Dictionary ...")
    return get_dictionary(db.pdb, PdbChain, key_field)


def retrofit_ecod_wordvecs(inpath, outpath, th=0.95):
    nodes_dic = get_ecod_dictionary("uid")
    cluster_dic, reverse_dic = get_cdhit_clusters(th, ECOD)
    logger.info("Retrofitting...")
    lexicon = "%s/synonyms.%s.txt" % (ckptpath, th)
    ecods = [map(lambda uid: nodes_dic[uid].name, ids)
             for ids in cluster_dic.values()]
    with open(lexicon, 'w+') as f:
        lines = ["%s\n" % " ".join(synonyms) for synonyms in ecods]
        f.writelines(lines)
    os.system("python ../../../retrofitting/retrofit.py -i %s -l %s -n 10 -o %s"
              % (inpath, lexicon, outpath))
    return outpath


class Dictionary(object):

    def __init__(self, name=None):
        self._model = dict()
        self._name = name

    def load(self, emb):
        pass

    @property
    def name(self):
        return self._name

    @property
    def vectors(self):
        return self._model.values()

    def wordvecs(self, keys):
        return np.matrix([self[key] for key in keys])

    def similarity(self, w1, w2):
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def __getitem__(self, key):
        return self._model[key]

    def __setitem__(self, key, val):
        self._model[key] = val

    def __contains__(self, key):
        return key in self._model

    def __len__(self):
        return len(self._model)

    @property
    def vocab(self):
        return self._model.keys()


class Clstr2Vec(Dictionary):

    def __init__(self, name="Clstr2Vec"):
        super(Clstr2Vec, self).__init__(name)
        self.data = dict()

    def load(self, emb):
        if os.path.exists(emb):
            logger.info("Reading input Model from src=%s" % emb)
            with open(emb, 'r') as f:
                data = self.data = json.load(f)
                labels = OneHotEncoder().fit_transform([[val] for val in data.values()])
                self._model = {key: labels[i] for i, key in enumerate(data.keys())}
        else:
            logger.error("%s not found" % emb)

    def wordvecs(self, keys):
        data = {key: self.data[key] for key in keys}
        labels = OneHotEncoder().fit_transform([[val] for val in data.values()])
        sparse_vectors = {key: labels[i] for i, key in enumerate(data.keys())}
        return np.matrix([sparse_vectors[key].toarray()[0] for key in keys])

    def __getitem__(self, key):
        return self._model[key].toarray()[0]

    def __setitem__(self, key, val):
        self._model[key] = sparse.csr_matrix(val)


class Node2Vec(Dictionary):

    def __init__(self, name="Node2Vec"):
        super(Node2Vec, self).__init__(name)

    def load(self, emb):
        if os.path.exists(emb):
            logger.info("Reading input Model from src=%s" % emb)
            df = pd.read_csv(emb, header=None, skiprows=1, sep=" ")
            for i, row in df.iterrows():
                if row[0] in self:
                    tmp = np.zeros(emb_dim*2)
                    tmp[:emb_dim] = self[row[0]]
                    tmp[emb_dim:] = row[1:emb_dim + 1]
                    self[row[0]] = tmp
                else:
                    self[row[0]] = np.array(row[1:emb_dim + 1])
        else:
            logger.error("%s not found" % emb)

    def train(self, edgelist, emb):
        if os.path.exists(edgelist):
            logger.info("Reading input Graph from src=%s" % edgelist)
            G = nx.read_edgelist(edgelist, data=False)
            nodes, edges = G.nodes(), G.edges()
            translation_dict = {}
            for i in range(len(nodes)):
                nx.relabel_nodes(G, {nodes[i]: i}, copy=False)
                translation_dict[i] = nodes[i]
            nx.write_edgelist(G, '%s/translated.edgelist' % ckptpath, data=False)
            num_cc = nx.number_connected_components(G)
            logger.info("Training Node2Vec.: #nodes=%s, #edges=%s #CC=%s"
                        % (len(nodes), len(edges), num_cc))
            os.system("node2vec -i:%s/translated.edgelist -o:%s/translated.emb -d:%s -v"
                      % (ckptpath, ckptpath, emb_dim))
            df = pd.read_csv("%s/translated.emb" % ckptpath, header=None, skiprows=1, sep=" ")
            for i, row in df.iterrows():
                key = translation_dict[row[0]]
                df.loc[i, 0] = key
                self[key] = np.array(row[1:])
            df.to_csv(emb, sep=" ", index=False, header=None)
        else:
            logger.error("%s not found" % edgelist)


def create_uniprot_edgelist(perc, num_iter):
    dst = '%s/uniprot.%s.edgelist' % (ckptpath, perc)
    G = UniprotGraph(dst)
    G.coalesce(thicken, perc/100.0, num_iter)
    G.to_edgelist(dst)


def create_pdb_edgelist(perc=95):
    dst = '%s/pdb.%s.edgelist' % (ckptpath, perc)
    G = PdbGraph(dst)
    G.coalesce(thicken, perc/100.0)
    G.to_edgelist(dst)


def create_ecod_edgelist(perc=95):
    dst = '%s/ecod.%s.edgelist' % (ckptpath, perc)
    G = EcodGraph(dst)
    G.coalesce(thicken, perc/100.0)
    G.to_edgelist(dst)


class NGram2Vec(object):

    def __init__(self, opts):
        n = opts["ngram_size"]
        c = opts["win_size"]
        d = opts["dim_size"]
        mc = opts["min_count"]
        dbname = opts["db"]
        if not os.path.exists("%s/n-gram" % ckptpath):
            os.makedirs("%s/n-gram" % ckptpath)
        unique_str = "%s-gram:%s:dim=%s:win=%s:mc=%s" % (n, dbname, d, c, mc)
        model_filename = "%s/n-gram/%s" % (ckptpath, unique_str)
        if os.path.exists(model_filename):
            self.model = Word2Vec.load(model_filename)
        else:
            sentences_path = "%s/n-gram/%s-gram:%s.sentences" % (ckptpath, n, dbname)
            sequences = map(lambda prot: prot.seq, map(Uniprot, db.uniprot.find({"db": dbname})))
            sentences = (" ".join(NGram2Vec.get_sequence_ngrams(seq, n, k))
                         for seq in sequences for k in range(n))
            mode = 'r' if os.path.exists(sentences_path) else 'w+'
            with open(sentences_path, mode) as f:
                if mode == 'w+': f.writelines(sentences)    # file was created
                src = LineSentence(f)
                self.model = Word2Vec(src,
                                      size=d,
                                      window=c,
                                      min_count=mc,
                                      workers=4, sg=1)
                self.model.save(model_filename)

    @staticmethod
    def get_sequence_ngrams(seq, n, offset=0):
        return (
            filter(
                lambda ngram: len(ngram) == n, [seq[i:min(i + n, len(seq))]
                                                for i in range(offset, len(seq), n)]
            )
        )

    def similarity(self, w1, w2):
        return self.model.similarity(w1, w2)

    def __getitem__(self, key):
        return np.array(self.model[key], dtype=np.float64)

    def __contains__(self, key):
        return key in self.model


class BagOfNGrams(object):

    def __init__(self, ngram_size, db="sp", dim_size=100, win_size=25, min_count=2,
                 collection=db.uniprot, Model=Uniprot):

        self.ngram_size = ngram_size
        self.dim_size = dim_size
        self.ngram_model = NGram2Vec({
            "ngram_size": ngram_size,
            "dim_size": dim_size,
            "win_size": win_size,
            "min_count": min_count,
            "db": db
        })
        self.collection = collection
        self.Model = Model
        self._name = "bag_of_%s-grams" % ngram_size

    @property
    def name(self):
        return self._name

    def __getitem__(self, seq):
        n, bow = self.ngram_size, np.zeros(self.dim_size)
        for k in range(n):
            ngrams = NGram2Vec.get_sequence_ngrams(seq, n, k)
            for ngram in filter(lambda ngram: ngram in self.ngram_model, ngrams):
                bow += self.ngram_model[ngram]
        return bow / len(seq)

    def __contains__(self, key):
        return True

    def wordvecs(self, keys):
        Model, collection = self.Model, self.collection
        data = map(Model, collection.find({"_id": {"$in": keys}}))
        return np.matrix([self[seq.seq] for seq in data])


def main():
    model3 = BagOfNGrams(3)
    # model4 = BagOfNGrams(4)


    # Node2Vec().train('%s/intact.all.edgelist' % ckptpath, "%s/intact.all.emb" % ckptpath)

    # retrofit_ecod_wordvecs("%s/ecod.simple.emb" % ckptpath, "%s/retrofitted.99.ecod.emb" % ckptpath, th=.99)

if __name__ == "__main__":
    main()
