import os
import sys
import subprocess

from tqdm import tqdm

from Bio.Seq import Seq
from Bio import SeqIO, SearchIO
from Bio.SeqRecord import SeqRecord

from Bio.Blast.Applications import NcbiblastpCommandline

from src.python.preprocess2 import *

from itertools import cycle
import matplotlib.pyplot as plt

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()

from concurrent.futures import ThreadPoolExecutor

import argparse


ASPECT = 'F'
ONTO = None
PRIOR = None
THRESHOLDS = np.arange(.05, 1, .05)

cleanup = True
eps = 10e-6


def init_GO(asp=ASPECT, src=None):
    global ONTO, ASPECT
    if src: set_obo_src(src)
    ASPECT = asp
    ONTO = get_ontology(asp)
    return ONTO


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")


def load_all_data():
    mf, _ = load_data(db, asp='F', codes=exp_codes)
    cc, _ = load_data(db, asp='C', codes=exp_codes)
    bp, _ = load_data(db, asp='P', codes=exp_codes)
    return mf, cc, bp


def _prepare_naive(reference):
    global PRIOR
    prior_pth = os.path.join(tmp_dir, 'prior-%s.npy' % GoAspect(ASPECT))
    if os.path.exists(prior_pth):
        PRIOR = np.load(prior_pth).item()
    go2count = {}
    for _, go_terms in reference.items():
        for go in go_terms:
            if go in go2count:
                go2count[go] += 1
            else:
                go2count[go] = 1
    total = len(reference)
    prior = {go: count/total for go, count in go2count.items()}
    np.save(prior_pth, prior)
    PRIOR = prior


def _naive(target, reference):
    global PRIOR
    if not PRIOR:
        _prepare_naive(reference)
    return PRIOR


def _prepare_blast(sequences):
    # print('### entering _prepare_blast')
    blastdb_pth = os.path.join(tmp_dir, 'blast-%s' % GoAspect(ASPECT))
    records = [SeqRecord(Seq(seq), id) for id, seq in sequences.items()]
    SeqIO.write(records, open(blastdb_pth, 'w+'), "fasta")
    os.system("makeblastdb -in %s -dbtype prot" % blastdb_pth)


def parallel_blast(targets, reference, num_cpu=4):
    blastdb_pth = os.path.join(tmp_dir, 'blast-%s' % GoAspect(ASPECT))
    records = [SeqRecord(Seq(seq), id) for id, seq in reference.items()]
    SeqIO.write(records, open(blastdb_pth, 'w+'), "fasta")
    os.system("makeblastdb -in %s -dbtype prot" % blastdb_pth)

    predictions = dict()
    e = ThreadPoolExecutor(num_cpu)

    def _parallel_blast_helper(s):
        return s[0], _blast(SeqRecord(Seq(s[1]), s[0]), reference, topn=None, choose_max_prob=True)

    pbar = tqdm(range(len(targets)), desc="blast2go processed")

    for tgtid, preds in e.map(_parallel_blast_helper, targets.items()):
        predictions[tgtid] = preds
        pbar.update(1)

    pbar.close()
    return predictions


def _blast(target_fasta, reference, topn=None, choose_max_prob=True):
    seqid, asp = target_fasta.id, GoAspect(ASPECT)
    query_pth = os.path.join(tmp_dir, "%s-%s.fas" % (seqid, asp))
    output_pth = os.path.join(tmp_dir, "%s-%s.out" % (seqid, asp))
    database_pth = os.path.join(tmp_dir, 'blast-%s' % asp)

    SeqIO.write(target_fasta, open(query_pth, 'w+'), "fasta")

    cline = NcbiblastpCommandline(query=query_pth, db=database_pth, out=output_pth,
                                  outfmt=5, evalue=0.001, remote=False, ungapped=False)

    child = subprocess.Popen(str(cline),
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=(sys.platform != "win32"))

    handle, _ = child.communicate()
    assert child.returncode == 0

    blast_qresult = SearchIO.read(output_pth, 'blast-xml')

    annotations = {}
    for hsp in blast_qresult.hsps[:topn]:

        if hsp.hit.id == seqid:
            continue

        ident = hsp.ident_num / hsp.hit_span

        for go in reference[hsp.hit.id]:
            if go in annotations:
                annotations[go].append(ident)
            else:
                annotations[go] = [ident]

    for go, ps in annotations.items():
        if choose_max_prob:
            annotations[go] = max(ps)
        else:
            annotations[go] = 1 - np.prod([(1 - p) for p in ps])

    if cleanup:
        os.remove(query_pth)
        os.remove(output_pth)

    return annotations


def _predict(reference_annots, target_seqs, func_predict, binary_mode=False):
    if len(target_seqs) > 1:
        pbar = tqdm(range(len(target_seqs)), desc="targets processed")
    else:
        pbar = None
    if binary_mode:
        predictions = np.zeros((len(target_seqs), len(ONTO.classes)))
        for i, (_, seq) in enumerate(target_seqs.items()):
            preds = func_predict(seq, reference_annots)
            bin_preds = ONTO.binarize([list(preds.keys())])[0]
            for go, prob in preds.items():
                bin_preds[ONTO[go]] = prob
            predictions[i, :] = bin_preds
            if pbar: pbar.update(1)
    else:
        predictions = {}
        for _, (seqid, seq) in enumerate(target_seqs.items()):
            predictions[seqid] = func_predict(SeqRecord(Seq(seq), seqid), reference_annots)
            if pbar: pbar.update(1)
    if pbar: pbar.close()
    return predictions


def bin2dict(distribution, classes):
    return {classes[i]: prob for i, prob in enumerate(distribution)}


def get_P_and_T_from_dictionaries(tau, predictions, targets):
    P, T = [], []
    for seqid, seq_preds in predictions.items():
        seq_targets = targets[seqid]
        assert len(seq_targets) > 0
        seq_annots = [go for go, prob in seq_preds.items() if prob >= tau]
        P.append(set(seq_annots))
        T.append(set(seq_targets))
    assert len(P) == len(T)
    return P, T


def get_P_and_T_from_arrays(tau, predictions, targets, classes):
    P, T = [], []
    for seq_preds, seq_targets in zip(map(lambda p: bin2dict(p, classes), predictions),
                                      map(lambda t: bin2dict(t, classes), targets)):
        seq_annots = [go for go, prob in seq_preds.items() if prob >= tau]
        seq_targets = [go for go, prob in seq_targets.items() if prob >= tau]
        assert len(seq_targets) > 0
        P.append(set(seq_annots))
        T.append(set(seq_targets))
    assert len(P) == len(T)
    return P, T


def precision(tau, predictions, targets, classes=None):
    assert type(predictions) == type(targets)
    if isinstance(predictions, dict):
        P, T = get_P_and_T_from_dictionaries(tau, predictions, targets)
    else:
        assert classes
        P, T = get_P_and_T_from_arrays(tau, predictions, targets, classes)
    ret = [len(P_i & T_i) / len(P_i) if len(P_i) else 1.0 for P_i, T_i in zip(P, T)]
    return ret


def recall(tau, predictions, targets, classes=None, partial_evaluation=False):
    assert type(predictions) == type(targets)
    if isinstance(predictions, dict):
        P, T = get_P_and_T_from_dictionaries(tau, predictions, targets)
    else:
        assert classes
        P, T = get_P_and_T_from_arrays(tau, predictions, targets, classes)
    if partial_evaluation:
        P, T = zip(*[(P_i, T_i) for P_i, T_i in zip(P, T) if len(P_i) > 0])
    ret = [len(P_i & T_i) / len(T_i) for P_i, T_i in zip(P, T)]
    return ret


def F_beta(pr, rc, beta=1):
    if pr == 0 and rc == 0: return eps
    return (1 + beta ** 2) * ((pr * rc) / (((beta ** 2) * pr) + rc))


def F1(pr, rc):
    if pr == 0 and rc == 0: return eps
    return 2 * ((pr * rc) / (pr + rc))


def predict(reference_seqs, reference_annots, target_seqs, method, basename=""):
    filename = "%s_%s.npy" % (method, basename)
    if method == "blast":
        pred_path = os.path.join(tmp_dir, filename)
        if basename and os.path.exists(pred_path):
            return np.load(pred_path).item()
        _prepare_blast(reference_seqs)
        predictions = _predict(reference_annots, target_seqs, _blast)
        np.save(pred_path, predictions)
        return predictions
    elif method == "naive":
        _prepare_naive(reference_annots)
        predictions = _predict(reference_annots, target_seqs, _naive)
        return predictions
    elif method == "deepseq":
        pred_path = os.path.join(tmp_dir, filename)
        return np.load(pred_path).item()
    elif method == "seq2go":
        pred_path = os.path.join(tmp_dir, filename)
        return np.load(pred_path).item()
    elif method == "seq2go-proba":
        pred_path = os.path.join(tmp_dir, filename)
        return np.load(pred_path).item()
    else:
        print("Unknown method")


def performance(predictions, ground_truth, classes=None, ths=THRESHOLDS):
    prs, rcs, f1s = [], [], []
    for tau in ths:
        pr_per_seq = precision(tau, predictions, ground_truth, classes)
        rc_per_seq = recall(tau, predictions, ground_truth, classes)
        pr_tau = np.mean(pr_per_seq)
        rc_tau = np.mean(rc_per_seq)
        prs.append(pr_tau)
        rcs.append(rc_tau)
        f1s.append(np.mean(F1(pr_tau, rc_tau)))
    return ths, prs, rcs, f1s


def plot_precision_recall(perf):
    # Plot Precision-Recall curve
    lw, n = 2, len(perf)
    methods = list(perf.keys())
    prs = [v[1] for v in perf.values()]
    rcs = [v[2] for v in perf.values()]
    f1s = [v[3] for v in perf.values()]

    colors = cycle(['red', 'blue', 'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    # Plot Precision-Recall curve for each class
    plt.clf()

    for i, color in zip(range(len(methods)), colors):
        plt.plot(rcs[i], prs[i], color=color, lw=lw,
                 label='{0} (F_max = {1:0.2f})'
                 .format(methods[i], max(f1s[i])))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(GoAspect(ASPECT))
    plt.legend(loc="lower right")
    plt.show()


def evaluate_performance(db, methods, asp='F', train_and_validation_data=None, filename=None, plot=1):
    onto = init_GO(asp)
    if train_and_validation_data:
        seqs_train, annots_train, seqs_valid, annots_valid = train_and_validation_data
    else:
        seqs_train, annots_train, seqs_valid, annots_valid = load_training_and_validation(db, None)
    annots_train = propagate(annots_train, onto, include_root=False)
    annots_valid = propagate(annots_valid, onto, include_root=False)
    perf = {}
    for meth in methods:
        pred = predict(seqs_train, annots_train, seqs_valid, meth, filename)
        perf[meth] = performance(pred, annots_valid)
    if plot == 1:
        plot_precision_recall(perf)
    return pred, perf


def product_of_experts(*predictions):

    def go2p2go2ps(go2p_arr):
        go2ps = dict()
        for go2p in go2p_arr:
            for go, prob in go2p.items():
                if go in go2ps:
                    go2ps[go].append(prob)
                else:
                    go2ps[go] = [prob]
        return go2ps

    poe = dict()
    for pred in predictions:
        for seqid, go2prob in pred.items():
            if seqid in poe:
                poe[seqid].append(pred[seqid])
            else:
                poe[seqid] = [pred[seqid]]

    for seqid, arr in poe.items():
        poe[seqid] = go2p2go2ps(arr)

    for seqid, go2prob in poe.items():
        for go, ps in go2prob.items():
            poe[seqid][go] = 1 - np.prod([(1 - p) for p in ps])

    return poe


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    lim = 100
    init_GO(ASPECT)

    t0 = datetime(2017, 1, 1, 0, 0)
    t1 = datetime.utcnow()

    seqs_train, annots_train, seqs_valid, annots_valid = load_training_and_validation(db, t0, t1, ASPECT, lim)

    predictions_blast = predict(seqs_train, annots_train, seqs_valid, "blast")
    ths, prs, rcs, f1s = performance(predictions_blast, annots_valid)

    import json
    print(json.dumps(predictions_blast, indent=1))
    print(json.dumps(annots_valid, indent=1))

    import pandas as pd
    print(pd.DataFrame({"Threshold": ths, "Precision": prs, "Recall": rcs, "F1": f1s}).head(20))
    print(len(seqs_train), len(seqs_valid), len(predictions_blast))
