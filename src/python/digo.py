from src.python.preprocess2 import *

from src.python.geneontology import *

from src.python.baselines import *

# from Bio import SeqIO
# from Bio.Seq import Seq as BioSeq
# from Bio.SeqRecord import SeqRecord

from scipy.stats import gaussian_kde

from concurrent.futures import ThreadPoolExecutor

import datetime

import pickle

from blast import *

NUM_CPU = 8

PRUNE_CONST = 3

eps = 10e-6

E = ThreadPoolExecutor(NUM_CPU)

np.random.seed(101)

tmp_dir = gettempdir()

EVAL = 10e5

verbose = False


def compute_node_prior(node, graph):
    node.prior = 0.5 + 0.5 * node.size / len(graph.sequences)


def get_metric(metric):
    cache = dict()

    def do_metric(seq1, seq2):
        key1 = (seq1.uid, seq2.uid)
        key2 = (seq2.uid, seq1.uid)
        if key1 in cache:
            val = cache[key1]
        elif key2 in cache:
            val = cache[key2]
        else:
            val = metric(seq1, seq2)
            cache[key1] = val
            cache[key2] = val
        return val

    return do_metric


def get_f(metric, agg):
    return lambda seq, sequences: agg([metric(seq, other) for other in sequences])


def is_seq_in_node(seq, node, use_prior=True):  # assuming f computed for seq and f_dist_in/out computed for node

    f_seq_node = seq.f[node]  # precompute f(seq, node)
    prior = node.prior if use_prior else 1.0
    prob_f_given_node = node.f_dist_in(f_seq_node)
    prior_prob_f_node = node.f_dist_out(f_seq_node)
    return prior * (prob_f_given_node / prior_prob_f_node)


class Node(object):

    def __init__(self, go, sequences, fathers, children):
        self.go = go
        self.sequences = sequences
        self.fathers = fathers
        self.children = children
        self._f_dist_out = None
        self._f_dist_in = None
        self._plus = None

    def __iter__(self):
        for seq in self.sequences:
            yield seq

    def __repr__(self):
        return "Node(%s, %d)" % (self.go, self.size)

    def __hash__(self):
        return hash(self.go)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.go == other.go
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return len(self.fathers) == 0

    @property
    def plus(self):
        if not self._plus:
            union = reduce(
                lambda s1, s2: s1 | s2,
                map(lambda c: c.sequences, self.children), set())
            assert len(union) <= self.size
            self._plus = list(self.sequences - union)
        return self._plus

    @property
    def size(self):
        return len(self.sequences)

    @property
    def f_dist_out(self):
        if self._f_dist_out:
            return self._f_dist_out
        else:
            raise(KeyError("f_dist_out not computed for %s" % self))

    @property
    def f_dist_in(self):
        if self._f_dist_in:
            return self._f_dist_in
        else:
            raise(KeyError("f_dist_in not computed for %s" % self))

    def sample(self, m):
        n = min(self.size, m)
        sequences = self.sequences
        s = set(np.random.choice(sequences, n, replace=False))
        assert len(s) == n > 0
        return s


def get_distribution(dataset):
    assert len(dataset) >= PRUNE_CONST
    return Distribution(dataset)


class Distribution(object):

    def __init__(self, dataset):
        self.kde = gaussian_kde([d * 10 for d in dataset])

    def __call__(self, *args, **kwargs):
        v = args[0]
        return self.kde.pdf(v)[0]


class Histogram(object):

    def __init__(self, dataset):
        self.bins = {(a, a + 1): .01 for a in range(10)}
        for p in dataset:
            a = min(int(p * 10), 9)
            self.bins[(a, a + 1)] += 0.9 / len(dataset)

    def __call__(self, *args, **kwargs):
        v = int(args[0] * 10)
        return self.bins[(v, v + 1)]


class Graph(object):
    def __init__(self, onto, uid2seq, go2ids):
        self.nodes = nodes = {}
        self.sequences = sequences = set()

        nodes[onto.root] = self.root = Node(onto.root, set(), [], [])

        for go, ids in go2ids.items():
            seqs = set([Seq(uid, uid2seq[uid]) for uid in ids])
            nodes[go] = Node(go, seqs, [], [])
            sequences |= seqs

        for go, obj in onto._graph._node.items():
            if 'is_a' not in obj:
                assert go == onto.root
                continue
            if go not in go2ids:
                assert go not in nodes
                continue
            if go not in nodes:
                assert go not in go2ids
                continue
            for father in obj['is_a']:
                nodes[go].fathers.append(nodes[father])
                nodes[father].children.append(nodes[go])

        for node in nodes.values():
            if node.is_leaf():
                assert node.size > 0
                continue
            children = node.children
            for child in children:
                assert child.size > 0
                node.sequences |= child.sequences

        for node in nodes.values():
            compute_node_prior(node, self)

    def predict_seq(self, seq, f):
        predictions = {}
        compute_f(f, seq, self)
        for node in self.nodes.values():
            score = is_seq_in_node(seq, node)
            predictions[node.go] = score
        return predictions

    def estimate_distributions(self, precomputed_sequences, attr_name):
        for node in graph:
            dataset = [seq.f[node] for seq in precomputed_sequences]
            setattr(self, attr_name, get_distribution(dataset))

    def prune(self, gte):
        to_be_deleted = []
        for go, node in self.nodes.items():
            if node.size >= gte:
                continue
            for father in node.fathers:
                father.children.remove(node)
            for child in node.children:
                child.fathers.remove(node)
            to_be_deleted.append(node)
        for node in to_be_deleted:
            del self.nodes[node.go]
        return to_be_deleted

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        for node in self.nodes.values():
            yield node

    def sample(self, max_add_to_sample=10):
        def sample_recursive(node, sampled):
            if not node.is_leaf():
                for child in node.children:
                    sampled |= sample_recursive(child, sampled)
            s = min(max_add_to_sample, len(node.plus))
            if len(node.plus) > 0:
                sampled |= set(np.random.choice(node.plus, s, replace=False))
            return sampled
        return sample_recursive(self.root, set())


def compute_f(f, seq, node, submitted, pbar):
    if not node.is_leaf():
        for child in node.children:
            task = compute_f(f, seq, child, submitted, pbar)
            if verbose: print("wait %s" % child)
            assert child in submitted
            val = task.result()
            seq.f[child] = val     # wait for children's results
            if verbose: print("finished %s f_val=%.2f" % (child, val))
    try:
        task = submitted[node]
    except KeyError:
        task = E.submit(f, seq, node.sequences - {seq})
        assert node not in submitted
        submitted[node] = task
        pbar.update(1)
    return task


def get_blast(blast):
    def do_blast(seq1, seq2):
        hits = blast[seq1]
        assert len(hits) > 0
        subjects = [h.sseqid for h in hits]
        try:
            ix = subjects.index(seq2.uid)
            return hits[ix].bitscore
        except ValueError:
            ix = np.argmin([h.bitscore for h in hits])
            return hits[ix].bitscore
    return do_blast


def blastp(seq1, seq2, evalue=EVAL):
    query_pth = os.path.join(tmp_dir, "%s.seq" % seq1.uid)
    subject_pth = os.path.join(tmp_dir, "%s.seq" % seq2.uid)
    output_pth = os.path.join(tmp_dir, "%s_%s.out" % (seq1.uid, seq2.uid))
    SeqIO.write(SeqRecord(BioSeq(seq1.seq), seq1.uid), open(query_pth, 'w+'), "fasta")
    SeqIO.write(SeqRecord(BioSeq(seq2.seq), seq2.uid), open(subject_pth, 'w+'), "fasta")
    cline = "blastp -query %s -subject %s -outfmt 6 -out %s -evalue %d 1>/dev/null 2>&1" \
            % (query_pth, subject_pth, output_pth, evalue)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    assert os.path.exists(output_pth)
    with open(output_pth, 'r') as f:
        hits = [HSP(line.split('\t')) for line in f.readlines()]
        if len(hits) == 0:
            return eps
        hsp = hits[np.argmin([h.evalue for h in hits])]
        db.blast.update_one({"_id": hsp.uid}, {"$set": vars(hsp)}, upsert=True)
    return hsp.bitscore


def cleanup():
    os.remove("%s/*.seq" % tmp_dir)
    os.remove("%s/*.out" % tmp_dir)


class Blast(object):

    def __init__(self, nodes):

        self.sequences = reduce(lambda s1, s2: s1 | s2, map(lambda n: n.sequences, nodes), set())


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def run_compute_f(f, seqs):
    for i, seq in enumerate(seqs):
        pbar = tqdm(range(len(graph)), desc="[%s] (%d/%d) nodes processed" % (seq.uid, i + 1, len(seqs)))
        root_task = compute_f(f, seq, graph.root, {}, pbar=pbar)
        root_task.result()   # wait for all other tasks to finish
        pbar.close()


if __name__ == "__main__":
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prot2vec']

    asp = 'F'   # molecular function
    onto = get_ontology(asp)
    t0 = datetime.datetime(2014, 1, 1, 0, 0)
    t1 = datetime.datetime(2014, 9, 1, 0, 0)

    print("Indexing Data...")
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, asp)
    print("Loading Training Data...")
    uid2seq_trn, _, go2ids_trn = trn_stream.to_dictionaries(propagate=True)
    print("Loading Validation Data...")
    uid2seq_tst, _, go2ids_tst = tst_stream.to_dictionaries(propagate=True)

    print("Building Graph...")
    graph = Graph(onto, uid2seq_trn, go2ids_trn)
    print("Graph contains %d nodes" % len(graph))

    print("Pruning Graph...")
    deleted_nodes = graph.prune(PRUNE_CONST)
    print("Pruned %d, Graph contains %d" % (len(deleted_nodes), len(graph)))

    blast_precomp = BLAST(db.blast)
    f = get_f(get_metric(get_blast(blast_precomp)), agg=np.mean)

    print("Computing f \"Inside\"")
    sample_of_inside = graph.sample(max_add_to_sample=3)
    blast_precomp.load_precomputed(sample_of_inside)
    run_compute_f(f, sample_of_inside)
    graph.estimate_distributions(sample_of_inside, "_f_dist_in")
    cleanup()

    save_object(graph, "Data/digo-%s-graph")

    print("Computing f \"Outside\"")
    sample_of_nature = []
    nature_sequences = load_nature_repr_set(db)
    sample_of_nature = np.random.choice(nature_sequences, 1000, replace=False)
    blast_precomp.load_precomputed(sample_of_nature)
    run_compute_f(f, sample_of_nature)
    graph.estimate_distributions(sample_of_nature, "_f_dist_out")
    cleanup()

    save_object(graph, "Data/digo-%s-graph")

    print("123 Predict...")
    seq_predictions = {}
    targets = [Seq(uid, seq) for uid, seq in uid2seq_tst.items()]
    blast_precomp.load_precomputed(targets)
    for tgt in targets:
        seq_predictions[tgt.uid] = graph.predict_seq(Seq(tgt.uid, seq), f)
    save_object(seq_predictions, "Data/digo_%s_preds_with_prior" % asp)
