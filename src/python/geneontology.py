import os
import obonet
import numpy as np
import networkx as nx

from sklearn.preprocessing import MultiLabelBinarizer

from gensim.models.poincare import PoincareModel, PoincareKeyedVectors

go_graph = None
mfo, cco, bpo = None, None, None

dim = 300
num_epochs = 5

# obo_src = "http://purl.obolibrary.org/obo/go/go-basic.obo"
obo_src = "Data/go-basic.obo"


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


def embedding(onto_graph, emb_fname):
    is_a = onto_graph.edges()
    model = PoincareModel(train_data=is_a, size=dim)
    model.train(epochs=num_epochs, print_every=500)
    model.kv.save(emb_fname)
    return model.kv


def get_ontology_graph(namespace):
    onto_graph = go_graph.copy()
    for n, attr in go_graph._node.items():
        if attr['namespace'] != namespace:
            onto_graph.remove_node(n)
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

        key_val = zip(self.classes, list(range(len(G))))
        self._direct_dict = {k: v for k, v in key_val}
        self._reverse_dict = {v: k for k, v in key_val}

        emb_fname = 'Data/%s-poincare-dim%d-epochs%d.emb' \
                    % (aspect, dim, num_epochs)

        if os.path.exists(emb_fname):
            self._kv = PoincareKeyedVectors.load(emb_fname)
        else:
            self._kv = embedding(G, emb_fname)

    @property
    def classes(self):
        return self._mlb.classes_

    def binarize(self, labels):
        return self._mlb.transform(labels)

    def todense(self, go_terms):
        return np.array([self._kv[go] for go in go_terms])

    def __getitem__(self, go):
        return self._direct_dict[go]

    def __str__(self):
        k, n = (len(nx.dag_longest_path(self._graph))), len(self)
        return "%s\n#GO-terms\t:\t%d\nmax-path\t:\t%d" % \
               (self._aspect, n, k)

    def __len__(self):
        return len(self.classes)
