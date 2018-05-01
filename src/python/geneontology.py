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

# obo_src = "http://purl.obolibrary.org/obo/go/go-basic.obo"
obo_src = "Data/go-basic.obo" if os.path.exists("Data") else "../../Data/go-basic.obo"

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
        classes = list(reversed(list(nx.topological_sort(G))))

        self.root = root = classes[0]
        self._levels = self.bfs(root)

        self._levels = levels = dict()
        for node, lvl in nx.shortest_path_length(G, target=root).items():
            if lvl in levels:
                levels[lvl].append(node)
            else:
                levels[lvl] = [node]

        self._mlb = MultiLabelBinarizer().fit([classes])

        key_val = [(go, i) for i, go in enumerate(classes)]
        self.go2ix = {k: v for k, v in key_val}
        self.ix2go = {v: k for k, v in key_val}

        emb_fname = os.path.join('%s/%s-poincare-dim%d-epochs%d.emb' % (DATA_ROOT, aspect, dim, num_epochs))

        if os.path.exists(emb_fname):
            self._kv = PoincareKeyedVectors.load(emb_fname)
        else:
            self._kv = embedding(ns, emb_fname)

    @property
    def classes(self):
        return [c for c in self._mlb.classes_]

    @property
    def num_levels(self):
        return len(self._levels)

    def sort(self, go_terms):
        return sorted(go_terms, key=lambda go: self[go])

    # If father to one of the leaves return True else False
    def is_father(self, father, leaves):
        d = nx.shortest_path_length(self._graph, target=father)
        for leaf in leaves:
            if leaf in d:
                return True
        return False

    def negative_sample(self, leaves, classes=None):
        if not classes: classes = self.classes
        can = np.random.choice(classes)
        while self.is_father(can, leaves):
            can = np.random.choice(classes)
        return can

    def propagate(self, go_terms, include_root=True):
        G = self._graph
        lbl = self.sort(filter(lambda x: G.has_node(x), go_terms))
        if include_root:
            anc = map(lambda go: self.sort(nx.descendants(G, go)) + [go], lbl)
        else:
            anc = map(lambda go: self.sort(nx.descendants(G, go))[1:] + [go], lbl)
        return reduce(lambda x, y: x + y, anc, [])

    def bfs(self, root):
        levels, G = dict(), self._graph
        for node, lvl in nx.shortest_path_length(G, target=root).items():
            if lvl in levels:
                levels[lvl].append(node)
            else:
                levels[lvl] = [node]
        return levels

    def get_level(self, lvl, root=None):
        if not root: return self._levels[lvl]
        else: return self.bfs(root)[lvl]

    def binarize(self, go_labels):
        return self._mlb.transform(go_labels)

    def todense(self, go):
        return self._kv[go]

    def __getitem__(self, go):
        return self.go2ix[go]

    def __str__(self):
        k, n = (len(nx.dag_longest_path(self._graph))), len(self)
        return "%s\n#GO-terms\t:\t%d\nmax-path\t:\t%d" % \
               (self._aspect, n, k)

    def __len__(self):
        return len(self.classes)


def least_common_ancestor(G, go1, go2):
    ancestors1 = nx.shortest_path_length(G, source=go1)
    ancestors2 = nx.shortest_path_length(G, source=go2)
    common = list(set(ancestors1.keys()) & set(ancestors2.keys()))
    assert len(common) > 0
    c = common[np.argmin([ancestors1[c] + ancestors2[c] for c in common])]
    return c, ancestors1[c] + ancestors2[c]


if __name__=="__main__":
    onto = get_ontology('P')
    print(onto.sort(["GO:0065007",
                     "GO:0008150",
                     "GO:0008152",
                     "GO:0009987",
                     "GO:0006807"]))
