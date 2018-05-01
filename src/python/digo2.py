import torch

from src.python.preprocess2 import *

from src.python.geneontology import *

from src.python.baselines import *

from src.python.dingo_utils import *

from src.python.dingo_net2 import *

from tqdm import tqdm

import itertools


# def get_metric(metric):
#     cache = dict()
#
#     def do_metric(seq1, seq2):
#         key1 = (seq1.uid, seq2.uid)
#         key2 = (seq2.uid, seq1.uid)
#         if key1 in cache:
#             val = cache[key1]
#         elif key2 in cache:
#             val = cache[key2]
#         else:
#             val = metric(seq1, seq2)
#             cache[key1] = val
#             cache[key2] = val
#         return val
#
#     return do_metric


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
    for node in graph:
        score = is_seq_in_node(seq, node)
        predictions[node.go] = score
    return predictions


def estimate_distributions(graph, precomputed_sequences, attr_name):
    for node in graph:
        dataset = [seq.f[node] for seq in precomputed_sequences]
        setattr(graph, attr_name, get_distribution(dataset))


def get_blast(blast, evalue):
    def do_blast(seq1, seq2):
        hits = blast.get_hits(seq1, seq2)
        if len(hits) > 0:
            hit = hits[np.argmin([h.evalue for h in hits])]
            return hit.bitscore
        else:
            hit = blast.blastp(seq1, seq2, evalue=evalue)
            return hit.bitscore
    return do_blast


def cleanup():
    files = os.listdir(tmp_dir)
    for file in files:
        if file.endswith(".seq") or file.endswith(".out"):
            os.remove(os.path.join(tmp_dir, file))


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


def predict_by_similarity(target_seq, nodes, metric, pbar, agg=np.mean):
    tasks = {}
    preds = {}
    for node in nodes:
        pbar.update(1)
        if target_seq not in node.seq2vec:
            continue
        vector = node.seq2vec[target_seq]
        vectors = [node.seq2vec[seq] for seq in node.sequences]
        task = E.submit(metric, vector, vectors)
        tasks[node] = task
    for node, task in tasks.items():
        cos_similarity_arr = task.result()
        assert len(cos_similarity_arr) > 0
        preds[node] = agg(cos_similarity_arr)
    return preds


def predict_by_ks(target_seq, nodes, metric, pbar):
    tasks = {}
    preds = {}
    for node in nodes:
        pbar.update(1)
        if target_seq not in node.seq2vec:
            continue
        vector = node.seq2vec[target_seq]
        vectors = [node.seq2vec[seq] for seq in node.sequences]
        task = E.submit(metric, vector, vectors)
        tasks[node] = task
    for node, task in tasks.items():
        cos_similarity_arr = task.result()
        assert len(cos_similarity_arr) > 0
        _, alpha = ks_2samp(node.dataset, cos_similarity_arr)
        preds[node] = alpha
    return preds


def preds_by_attr(hits_per_uid, attr, nb=None):
    preds = {}
    pbar = tqdm(range(len(hits_per_uid)), desc="sequences processed")
    for uid, hits in hits_per_uid.items():
        pbar.update(1)
        preds[uid] = {}
        if len(hits) == 0:
            continue
        for go, hits in hits.items():
            assert go != graph.root.go
            hs = [getattr(h, attr) for h in hits if h.evalue < 0.001]
            if len(hs) == 0:
                continue
            if nb:
                preds[uid][go] = nb.infer(max(hs), graph[go].prior)
            else:
                preds[uid][go] = max(hs)
    pbar.close()
    return preds


def propagate_leaf_predictions(leaf_predictions, choose_max_prob=False):
    node2probs = {}
    predictions = {}
    for leaf, prob in leaf_predictions.items():
        ancestors = propagate(leaf)
        for node in ancestors:
            if node in node2probs:
                node2probs[node].append(prob)
            else:
                node2probs[node] = [prob]
    for node, probs in node2probs.items():
        if choose_max_prob:
            predictions[node.go] = max(probs)
        else:
            predictions[node.go] = 1 - np.prod([1 - pr for pr in probs])
    return predictions


def compute_datasets(metric, nodes):
    all_tasks = {}
    for node in nodes:
        for seq in node.sequences:
            vec = node.seq2vec[seq]
            vectors = [node.seq2vec[s] for s in node.sequences - {seq}]
            task = E.submit(metric, vec, vectors)
            if node in all_tasks:
                all_tasks[node].append(task)
            else:
                all_tasks[node] = [task]
    assert len(all_tasks) == len(leaves)
    pbar = tqdm(range(len(leaves)), desc="leaves processed")
    for node, tasks in all_tasks.items():
        node.dataset = []
        for task in tasks:
            results = task.result()
            node.dataset.extend(results)
        assert len(node.dataset) > 0
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":

    cleanup()
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
    uid2seq_trn, uid2go_trn, go2uid_trn = trn_stream.to_dictionaries(propagate=True)
    print("Loading Validation Data...")
    uid2seq_tst, uid2go_tst, _ = tst_stream.to_dictionaries(propagate=True)

    print("Building Graph...")
    graph = Graph(onto, uid2seq_trn, go2uid_trn)
    print("Graph contains %d nodes" % len(graph))

    print("Pruning Graph...")
    deleted_nodes = graph.prune(3)
    print("Pruned %d, Graph contains %d" % (len(deleted_nodes), len(graph)))
    save_object(graph, "Data/dingo_%s_graph" % asp)

    print("Load DingoNet")
    go_embedding_weights = np.asarray([onto.todense(go) for go in onto.classes])
    net = DingoNet(ATTN, 100, 10, go_embedding_weights)
    net = net.cuda()
    ckpth = "/tmp/dingo_0.10420.tar"
    print("=> loading checkpoint '%s'" % ckpth)
    checkpoint = torch.load(ckpth, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])

    print("Precomputing graph vectors")
    leaves = graph.leaves
    nodes_data = [(seq, leaf) for leaf in leaves for seq in leaf.sequences]
    compute_vectors(nodes_data, net, onto)

    print("Compute K-S datasets")
    compute_datasets(fast_cosine_similarity, leaves)

    limit = None
    evalue = 0.001
    print("Running BLAST evalue=%s..." % evalue)
    targets = [Seq(uid, seq) for uid, seq in uid2seq_tst.items()][:limit]
    db_pth = prepare_blast(uid2seq_trn)
    hits_per_uid = predict_blast_parallel(targets, uid2go_trn, db_pth, evalue)
    predictions_pindent = preds_by_attr(hits_per_uid, "pident")
    save_object(hits_per_uid, "%s/blast_%s_%s_hsp" % (out_dir, evalue, GoAspect(asp)))
    save_object(predictions_pindent, "Data/blast_%s_preds" % (GoAspect(asp),))
    cleanup()

    print("Precomputing target vectors...")
    tgtid2nodes = {tgt: [graph[go] for go in terms if go in graph] for tgt, terms in hits_per_uid.items()}
    targets_data = [(tgt, node) for tgt in targets for node in tgtid2nodes[tgt.uid] if node.is_leaf()]
    compute_vectors(targets_data, net, onto)

    print("123 Predict...")
    dingo_predictions = {}
    blast_predictions = {}
    for i, tgt in enumerate(targets):
        msg = "[%d/%d] (%s) leaves processed" % (i, len(targets), tgt.uid)
        candidates = tgtid2nodes[tgt.uid]
        pbar = tqdm(range(len(candidates)), desc=msg)
        leaf_predictions = predict_by_ks(tgt, candidates, fast_cosine_similarity, pbar)
        predictions = propagate_leaf_predictions(leaf_predictions)
        dingo_predictions[tgt.uid] = predictions
        ths, _, _, f1s = performance({tgt.uid: predictions}, {tgt.uid: uid2go_tst[tgt.uid]})
        j = np.argmax(f1s)
        msg = "[%d/%d] (%s) F_max=%.2f @ tau=%.2f" % (i, len(targets), tgt.uid, f1s[j], ths[j])
        pbar.set_description(msg)
        pbar.close()
    save_object(dingo_predictions, "Data/dingo_%s_preds" % (GoAspect(asp),))
    cleanup()
