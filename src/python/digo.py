from src.python.preprocess2 import *

from src.python.geneontology import *

from src.python.baselines import *

from src.python.digo_utils import *

from tqdm import tqdm

import itertools


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


def compute_f_inside(f, seq, node, submitted, pbar):
    if not node.is_leaf():
        for child in node.children:
            if seq not in child.sequences:
                continue
            task = compute_f_inside(f, seq, child, submitted, pbar)
            if verbose: print("wait %s" % child)
            assert child in submitted
            val = task.result()
            seq.f[child] = val     # wait for children's results
            if verbose: print("finished %s f_val=%.2f" % (child, val))
    try:
        task = submitted[node]
    except KeyError:
        assert seq in node.sequences
        task = E.submit(f, seq, node.sequences - {seq})
        assert node not in submitted
        submitted[node] = task
        if pbar: pbar.update(1)
    return task


def compute_leaf_datasets(metric, leaves):
    tasks = {}
    for leaf in leaves:
        assert leaf.is_leaf()
        s_in = min(10, leaf.size)
        sample_in = np.random.choice(list(leaf.sequences), s_in, replace=False)
        pairs = set(itertools.combinations(sample_in, 2))
        task = E.submit(run_metric_on_pairs, metric, pairs, verbose=False)
        tasks[leaf] = task
    assert len(tasks) == len(leaves)
    pbar = tqdm(range(len(leaves)), desc="leaves processed")
    for node, task in tasks.items():
        node.dataset = task.result()
        assert len(node.dataset) > 0
        pbar.update(1)
    pbar.close()


def compute_f_outside(f, seq, node, submitted, pbar):
    if not node.is_leaf():
        for child in node.children:
            task = compute_f_outside(f, seq, child, submitted, pbar)
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
        if pbar: pbar.update(1)
    return task


def predict_seq(graph, seq, f):
    predictions = {}
    run_compute_f(f, [seq], graph, compute_f_outside)
    for node in graph.nodes.values():
        score = is_seq_in_node(seq, node)
        predictions[node.go] = score
    return predictions


def estimate_distributions(graph, precomputed_sequences, attr_name):
    for node in graph:
        dataset = [seq.f[node] for seq in precomputed_sequences]
        setattr(graph, attr_name, get_distribution(dataset))


def get_blast(blast):
    def do_blast(seq1, seq2):
        hits = blast.get_hits(seq1, seq2)
        if len(hits) > 0:
            hit = hits[np.argmin([h.evalue for h in hits])]
            return hit.bitscore
        else:
            hit = blastp(seq1, seq2)
            return hit.bitscore
    return do_blast


def cleanup():
    try:
        os.remove("%s/*.seq" % tmp_dir)
        os.remove("%s/*.out" % tmp_dir)
    except FileNotFoundError:
        pass


def run_compute_f(f, seqs, g, method):  # method in [compute_f_outside, compute_f_inside]
    # pbar = tqdm(range(len(seqs)), desc="sequences processed")
    for i, seq in enumerate(seqs):
        pbar = tqdm(range(len(g)), desc="[%s] (%d/%d) nodes processed" % (seq.uid, i + 1, len(seqs)))
        root_task = method(f, seq, g.root, {}, pbar=pbar)
        root_task.result()   # wait for all other tasks to finish
        # pbar.update(1)
        pbar.close()
    # pbar.close()


def propagate(leaf, include_root=False):
        Q = [leaf]
        visited = {leaf}
        while Q:
            node = Q.pop()
            for father in node.fathers:
                if not include_root and father.is_root():
                    continue
                if father not in visited:
                    visited.add(father)
                    Q.append(father)
        return visited


def get_leaves(node_set):
    leaf_set = set()
    prop_set = set()
    for node in node_set:
        prop_set |= propagate(node)
    for node in node_set:
        if node.is_leaf():
            leaf_set.add(node)
        else:
            children = set(node.children)
            if len(prop_set & children) == 0:
                leaf_set.add(node)
    return leaf_set


if __name__ == "__main__":
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prot2vec']

    asp = 'F'   # molecular function
    onto = get_ontology(asp)
    t0 = datetime.datetime(2014, 1, 1, 0, 0)
    t1 = datetime.datetime(2014, 9, 1, 0, 0)
    # t0 = datetime.datetime(2017, 1, 1, 0, 0)
    # t1 = datetime.datetime.utcnow()

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
    deleted_nodes = graph.prune(3)
    print("Pruned %d, Graph contains %d" % (len(deleted_nodes), len(graph)))
    save_object(graph, "Data/digo_%s_graph" % asp)

    blast_precomp = BLAST(db.blast)
    blast_metric = get_metric(get_blast(blast_precomp))
    f = get_f(blast_metric, agg=np.max)

    # print("Computing f \"Outside\"")
    # nature_sequences = load_nature_repr_set(db)
    # sample_of_nature = blast_precomp.sort_by_count(nature_sequences)[:1000]
    # blast_precomp.load_precomputed(sample_of_nature)
    # run_compute_f(f, sample_of_nature, graph, compute_f_outside)
    # graph.estimate_distributions(sample_of_nature, "_f_dist_out")
    # save_object(graph, "Data/digo-%s-graph")
    # cleanup()

    # print("Computing f \"Inside\"")
    # pth = "Data/digo_%s_sample_of_inside" % asp
    # if os.path.exists(pth):
    #     sample_of_inside = load_object(pth)
    # else:
    #     sample_of_inside = graph.sample(max_add_to_sample=2)
    #     save_object(sample_of_inside, pth)
    # blast_precomp.load_precomputed(sample_of_inside)
    # run_compute_f(f, sample_of_inside, graph, compute_f_inside)
    # graph.estimate_distributions(sample_of_inside, "_f_dist_in")
    # save_object(graph, "Data/digo_%s_graph")
    # cleanup()

    print("Computing KS datasets")
    leaves = [node for node in graph if node.is_leaf()]
    compute_leaf_datasets(blast_metric, leaves)
    save_object(graph, "Data/digo_%s_graph" % asp)

    # print("123 Predict...")
    # seq_predictions = {}
    # targets = [Seq(uid, seq) for uid, seq in uid2seq_tst.items()]
    # # blast_precomp.load_precomputed(targets)
    # for tgt in targets:
    #     seq_predictions[tgt.uid] = graph.predict_seq(tgt, f)
    # save_object(seq_predictions, "Data/digo_%s_preds_with_prior" % asp)
