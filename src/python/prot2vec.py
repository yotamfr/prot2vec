#!/usr/bin/env python

from gensim import matutils
from gensim.models import doc2vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Word2Vec
from pymongo import MongoClient
from tqdm import tqdm

from ecod import EcodDomain

assert doc2vec.FAST_VERSION > -1

from numpy import dot

from sklearn.decomposition import PCA

import itertools
import networkx as nx
import pandas as pd
import numpy as np
import math
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


def get_ecod_clusters(ratio=0.90):

    fasta_filename = ECOD

    cluster_filename = "%s.%s" % (fasta_filename, ratio)

    if not os.path.exists(cluster_filename):
        os.system("cdhit -i %s -o %s -c %s" % (fasta_filename, cluster_filename, ratio))

    # open the cluster file and set the output dictionary
    cluster_file, cluster_dic, reverse_dic = open("%s.clstr" % cluster_filename), {}, {}

    logger.info("Reading cluster groups...")
    # parse through the cluster file and store the cluster name + sequences in the dictionary
    cluster_groups = (x[1] for x in itertools.groupby(cluster_file, key=lambda line: line[0] == '>'))
    for cluster in cluster_groups:
        name = int(next(cluster).strip().split()[-1])
        ids = [seq.split('>')[1].split('...')[0].split('|')[0] for seq in next(cluster_groups)]
        cluster_dic[name] = ids
    for cluster, ids in cluster_dic.items():
        for seqid in ids:
            reverse_dic[seqid] = cluster
    logger.info("Detected %s cluster groups..." % len(cluster_dic))

    return cluster_dic, reverse_dic


class EcodGraph(object):

    def __init__(self, edgelist=None):
        self.edgelist = edgelist
        if edgelist and os.path.exists(edgelist):
            self.from_edgelist(edgelist)
        else:
            self.graph = nx.Graph()

    def init_from_collection(self):
        logger.info(self)
        logger.info("Initiating Graph from DB collection")
        nodes = list(map(EcodDomain, db.ecod.find({})))
        for i in tqdm(range(len(nodes)), desc="Nodes Processed"):
            u = nodes[i]
            for v in u.get_adj_nodes():
                self.add_edge(u.name, v.name)
        logger.info("Finished!")
        logger.info(self)

    def coalesce(self, func, th=.90, num_trials=2000):
        cluster_dic, reverse_dic = get_ecod_clusters(th)
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
        return 'G=(V,E) |E|=%s, |V|=%s #CCs=%s' %\
               (len(self.edges()), len(self.nodes()), self.number_connected_components())


class InterfaceGraph(nx.Graph):
    def __init__(self, db_collection):
        super(InterfaceGraph, self).__init__()
        self.collection = db_collection
        self.nodes_dic = {}

    def init_from_csv(self, csv_filename):
        df = pd.read_csv(csv_filename)
        for _, row in df.iterrows():
            if not (row['MinDist:Atom-Atom'] and row['MinDist:Calpha-Calpha']):
                continue
            self.add_node(row['FROM_MOTIF'], label=row['FROM_MOTIF'],
                          graphics=consts["NODE_GRAPHICS"]())
            self.add_node(row['TO_MOTIF'], label=row['TO_MOTIF'],
                          graphics=consts["NODE_GRAPHICS"]())
            self.add_edge(row['FROM_MOTIF'], row['TO_MOTIF'],
                          graphics=consts["EDGE_GRAPHICS"]())

    def init_from_collection(self, db_collection, Model):
        logger.info("Initiating Graph from DB collection")
        nodes = list(map(Model, db_collection.find({})))
        for i in tqdm(range(len(nodes)), desc="Nodes Processed"):
            u = nodes[i]
            self.nodes_dic[u.name] = u
            for v in u.get_adj_nodes():
                if not (self.has_edge(u.name, v.name) or self.has_edge(v.name, u.name)):
                    self.add_edge(u.name, v.name)
        num_cc = nx.number_connected_components(self)
        logger.info('Finished building G=(V,E) |E|=%s, |V|=%s #CCs=%s' %
                    (len(self.edges()), len(self.nodes()), num_cc))

    # get representatives from all CCs
    def representatives(self):
        representatives = {}
        n = float(len(self.nodes()))
        ccs = nx.connected_components(self)
        for cc in ccs:
            rep = np.random.choice(list(cc))
            representatives[rep] = len(cc) / n
        return representatives

    def coalesce(self, func, th=.90, max_trials=4000):
        nodes_dic = self.nodes_dic
        cluster_dic, reverse_dic = get_ecod_clusters(th)
        trial = 0
        while not nx.is_connected(self) and trial < max_trials:
            num_cc = nx.number_connected_components(self)
            args = (trial, len(self.edges()), len(self.nodes()), num_cc)
            logger.info('Coalescing trial:%s %s/%s CCs:%s' % args)
            ids = np.random.choice(self.nodes(), size=1000, replace=False)
            nodes = filter(lambda e: e.uid in reverse_dic,
                           map(lambda e: nodes_dic[e], ids))
            for u, v in itertools.combinations(nodes, 2):
                if reverse_dic[u.uid] == reverse_dic[v.uid]:
                    func(self, u.name, v.name)
            trial += 1

    def get_degrees(self):
        return np.array(self.degree(self.nodes()).values(), dtype=np.float)

    def sample_path(self, lmd):  # length of path is sampled from poisson dist.
        src = self.sample_node()
        nbs = self.neighbors(src)
        k = max(math.floor(np.random.poisson(lam=lmd)), 2)
        assert len(nbs) > 0  # no zero deg nodes
        if len(nbs) == 1:
            p = self.random_walk(src, k)
        else:
            k1 = int(k / 2)
            k2 = k - k1
            neighbor = self.sample_node(self.get_weights(nbs))
            p1 = self.random_walk(neighbor, k1, reverse=True, forbidden=[src])
            p2 = self.random_walk(src, k2, forbidden=p1)
            p = p1 + p2
        return list(map(lambda ns: np.random.choice(ns.split(',')), p)), src

    # sample node (weighted by degree)
    def sample_node(self, weights=None):
        if not weights:
            if not self.weights:
                self.weights = self.get_weights(self.nodes())
                a = self.a = list(self.weights.keys())
                w = self.w = list(self.weights.values())
            else:
                a = self.a
                w = self.w
        else:
            a = list(weights.keys())
            w = list(weights.values())
        return np.random.choice(a, p=w)

    def random_walk(self, src, k, reverse=False, forbidden=[], max_trials=10):
        if k < 1: return []
        curr = src
        path = [src]
        i = k
        while i > 1:
            nbs = self.neighbors(curr)
            next = self.sample_node(self.get_weights(nbs))
            trial = 0
            while (next in path) or (next in forbidden):
                trial += 1
                if trial > max_trials:
                    if reverse: path.reverse()
                    return path
                nbs = self.neighbors(curr)
                next = self.sample_node(self.get_weights(nbs))
            path.append(next)
            curr = next
            i -= 1
        assert len(path) == k
        if reverse: path.reverse()
        return path

    def to_gml(self, relpath):
        nx.write_gml(self, relpath)

    def get_weights(self, nodes):
        weights = {}
        degrees = self.degree(nodes)
        sigma = float(sum(degrees.values()))
        for v, w in degrees.items():
            weights[v] = w / sigma
        return weights


def generate_sample_sentences():
    logger.info("Generating Samples...")
    adjacency_list = create_adjacency_list()
    power_rule_sample(adjacency_list, "power_rule:all_nodes")


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


def get_ecod_dictionary(key_field):
    logger.info("Computing ECOD Dictionary...")
    nodes_dic = {}
    nodes = list(db.ecod.find({}))
    for i in tqdm(range(len(nodes)), desc="Docs Processed"):
        nodes_dic[nodes[i][key_field]] = EcodDomain(nodes[i])
    return nodes_dic


def retrofit_ecod_wordvecs(inpath, outpath, th=0.95):
    nodes_dic = get_ecod_dictionary("uid")
    cluster_dic, reverse_dic = get_ecod_clusters(th)
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


class Node2Vec(object):

    def __init__(self, edgelist, emb):
        self.model = {}
        if os.path.exists(emb):
            logger.info("Reading input Model from src=%s" % emb)
            df = pd.read_csv(emb, header=None, skiprows=1, sep=" ")
            for i, row in df.iterrows():
                self[row[0]] = np.array(row[1:emb_dim+1])
        elif os.path.exists(edgelist):
            logger.info("Reading input Graph from src=%s" % edgelist)
            G = nx.read_edgelist(edgelist, data=False)
            nodes, edges = G.nodes(), G.edges()
            translation_dict = {}
            for i in range(len(nodes)):
                nx.relabel_nodes(G, {nodes[i]: i}, copy=False)
                translation_dict[i] = nodes[i]
            nx.write_edgelist(G, '%s/translated.edgelist' % ckptpath, data=False)
            num_cc = nx.number_connected_components(G)
            logger.info("Training Node2Vec.: #nodes=%s, #edges=%s #CC=%s" % (len(nodes), len(edges), num_cc))
            os.system("node2vec -i:%s/translated.edgelist -o:%s/translated.emb -d:%s -v"
                      % (ckptpath, ckptpath, emb_dim))
            df = pd.read_csv("%s/translated.emb" % ckptpath, header=None, skiprows=1, sep=" ")
            for i, row in df.iterrows():
                key = translation_dict[row[0]]
                df.loc[i, 0] = key
                self[key] = np.array(row[1:])
            df.to_csv(emb, sep=" ", index=False, header=None)
        else:
            logger.error("Unknown method")

    def similarity(self, w1, w2):
        v1 = matutils.unitvec(self[w1])
        v2 = matutils.unitvec(self[w2])
        return dot(v1, v2)

    def __getitem__(self, key):
        return self.model[key.lower()]

    def __setitem__(self, key, val):
        self.model[key.lower()] = val


def main():

    src = '%s/ecod.simple.edgelist' % ckptpath
    G = EcodGraph(edgelist=src)
    G.to_edgelist(src)
    # G = EcodGraph()
    # G.init_from_collection()
    # G.to_edgelist(src)
    # G.coalesce(thicken, .95)
    # G.to_edgelist(src)

    model = Node2Vec('%s/ecod.simple.edgelist' % ckptpath, "%s/ecod.simple.emb" % ckptpath)

    retrofit_ecod_wordvecs("%s/ecod.simple.emb" % ckptpath, "%s/retrofitted.90.ecod.emb" % ckptpath, th=.90)

if __name__ == "__main__":
    main()
