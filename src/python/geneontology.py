import os
import sys
import obonet
import numpy as np
import networkx as nx

from functools import reduce

from tempfile import gettempdir

from sklearn.preprocessing import MultiLabelBinarizer

from gensim.models.poincare import PoincareModel, PoincareKeyedVectors

go_graph = None
mfo, cco, bpo = None, None, None

dim = 200
num_epochs = 3

DATA_ROOT = gettempdir()

obo_src = "http://purl.obolibrary.org/obo/go/go-basic.obo"
# obo_src = "Data/go-basic.obo"

verbose = True


def set_obo_src(src):
    global obo_src
    obo_src = src


def initialize_go():
    global go_graph
    if not go_graph: go_graph = obonet.read_obo(obo_src)
    assert nx.is_directed_acyclic_graph(go_graph)
    return go_graph


def get_ontology(aspect):

    initialize_go()

    global mfo, cco, bpo
    if aspect == 'F':
        mfo = Ontology('molecular_function')
        return mfo
    elif aspect == 'P':
        bpo = Ontology('biological_process')
        return bpo
    elif aspect == 'C':
        cco = Ontology('cellular_component')
        return cco
    else:
        print('Unknown ontology')


def embedding(namespace, emb_fname):
    graph = go_graph.copy()
    for n, attr in go_graph._node.items():
        if attr['namespace'] != namespace:
            graph.remove_node(n)
    model = PoincareModel(train_data=graph.edges(), size=dim)
    model.train(epochs=num_epochs, print_every=500)
    model.kv.save(emb_fname)
    return model.kv


def get_ontology_graph(namespace):
    onto_graph = go_graph.copy()
    for n, attr in go_graph._node.items():
        if attr['namespace'] != namespace:
            onto_graph.remove_node(n)
    for (u, v) in go_graph.edges():
        d = onto_graph.get_edge_data(u, v)
        if not d or 'is_a' not in d:
            if onto_graph.has_edge(u, v):
                onto_graph.remove_edge(u, v)
    return onto_graph


class GoAspect(object):
    def __init__(self, aspect=None):
        self._aspect = aspect

    @property
    def aspect(self):
        return self._aspect if self._aspect else 'unspecified'

    def __eq__(self, other):
        return (str(self) == str(other)) or (self.aspect == other)

    def __str__(self):
        aspect = self._aspect
        return "BPO" if aspect == 'P' or aspect == 'biological_process' \
            else "MFO" if aspect == 'F' or aspect == 'molecular_function' \
            else "CCO" if aspect == 'C' or aspect == 'cellular_component' \
            else "unspecified" if not aspect \
            else "unknown"


class Ontology(object):

    def __init__(self, ns):
        """
        :param ns: ['molecular_function', 'biological_process', 'cellular_component']

        """

        self._aspect = aspect = GoAspect(ns)
        self._graph = G = get_ontology_graph(ns)
        classes = list(nx.topological_sort(G))

        self._mlb = MultiLabelBinarizer().fit([classes])

        key_val = [(go, i) for i, go in enumerate(classes)]
        self._direct_dict = {k: v for k, v in key_val}
        self._reverse_dict = {v: k for k, v in key_val}

        emb_fname = os.path.join('%s/%s-poincare-dim%d-epochs%d.emb'
                                 % (DATA_ROOT, aspect, dim, num_epochs)) \


        if os.path.exists(emb_fname):
            self._kv = PoincareKeyedVectors.load(emb_fname)
        else:
            self._kv = embedding(ns, emb_fname)

    @property
    def classes(self):
        return self._mlb.classes_

    def sort(self, go_terms):
        return sorted(go_terms, key=lambda go: self[go])

    def augment(self, go_terms, max_length=None):
        G = self._graph
        lbl = set(filter(lambda x: G.has_node(x), go_terms))
        if max_length:
            anc = map(lambda go: nx.shortest_path_length(G, source=go), lbl)
            aug = set([k for d in anc for k, v in d.items() if v <= max_length])
        else:
            anc = map(lambda go: nx.descendants(G, go), lbl)
            aug = reduce(lambda x, y: x | y, anc, lbl)
        return aug

    def binarize(self, go_labels):
        return self._mlb.transform(go_labels)

    def todense(self, go):
        return self._kv[go]

    def __getitem__(self, go):
        return self._direct_dict[go]

    def __str__(self):
        k, n = (len(nx.dag_longest_path(self._graph))), len(self)
        return "%s\n#GO-terms\t:\t%d\nmax-path\t:\t%d" % \
               (self._aspect, n, k)

    def __len__(self):
        return len(self.classes)


if __name__=="__main__":
    onto = get_ontology('P')
    print(onto.sort(["GO:0065007",
                     "GO:0008150",
                     "GO:0008152",
                     "GO:0009987",
                     "GO:0006807"]))
