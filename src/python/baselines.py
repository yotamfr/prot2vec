import os
import sys
import subprocess

from tqdm import tqdm

from Bio.Seq import Seq
from Bio import SeqIO, SearchIO
from Bio.SeqRecord import SeqRecord

from Bio.Blast.Applications import NcbiblastpCommandline

from .consts import *

from .preprocess import *

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
THRESHOLDS = np.arange(0.1, 1, 0.02)

cleanup = True

# unparseables = ["Cross_product_review", "Involved_in",
#                 "gocheck_do_not_annotate",
#                 "Term not to be used for direct annotation",
#                 "gocheck_do_not_manually_annotate",
#                 "Term not to be used for direct manual annotation",
#                 "goslim_aspergillus", "Aspergillus GO slim",
#                 "goslim_candida", "Candida GO slim",
#                 "goslim_generic", "Generic GO slim",
#                 "goslim_metagenomics", "Metagenomics GO slim",
#                 "goslim_pir", "PIR GO slim",
#                 "goslim_plant", "Plant GO slim",
#                 "goslim_pombe", "Fission yeast GO slim",
#                 "goslim_yeast", "Yeast GO slim",
#                 "gosubset_prok", "Prokaryotic GO subset",
#                 "mf_needs_review", "Catalytic activity terms in need of attention",
#                 "termgenie_unvetted",
#                 "Terms created by TermGenie that do not follow a template and require additional vetting by editors",
#                 "virus_checked", "Viral overhaul terms"]


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


# def load_evaluation_data(setting="cafa2"):
#     if setting == "cafa2":
#         data_root = "data/cafa/CAFA2Supplementary_data/data/"
#
#         go_pth = os.path.join(data_root, "ontology", "go_20130615-termdb.obo")
#         go_copy_pth = "%s.copy" % go_pth
#
#         with open(go_pth, "rt") as fin:
#             with open(go_copy_pth, "wt") as fout:
#                 for line in fin:
#                     for term in unparseables:
#                         line = line.replace(term, '')
#
#                     fout.write(line)
#
#         init_GO(ASPECT, go_copy_pth)
#         fpath = os.path.join(data_root, "GO-t0", "goa.go.%s" % GoAspect(ASPECT))
#         num_mapping = count_lines(fpath, sep=bytes('\n', 'utf8'))
#         src_mapping = open(fpath, 'r')
#         ref2go, _ = MappingFileLoader(src_mapping, num_mapping).load()
#
#         trg2seq = dict()
#         for domain in ["archaea", "bacteria", "eukarya"]:
#             targets_dir = os.path.join(data_root, "CAFA2-targets", domain)
#             trg2seq.update(load_cafa2_targets(targets_dir))
#         trg2go = dict()
#         annots_dir = os.path.join(data_root, "benchmark", "groundtruth")
#         fpath = os.path.join(annots_dir, "propagated_%s.txt" % GoAspect(ASPECT))
#         num_mapping = count_lines(fpath, sep=bytes('\n', 'utf8'))
#         src_mapping = open(fpath, 'r')
#         d1, _ = MappingFileLoader(src_mapping, num_mapping).load()
#         trg2go.update(d1)
#         return trg2seq, trg2go
#     elif setting == "cafa3":
#         pass
#
#     else:
#         print("Unknown evaluation setting")
#     pass


# def load_cafa2_targets(targets_dir):
#
#     trg2seq = dict()
#
#     for fname in os.listdir(targets_dir):
#         print("\nLoading: %s" % fname)
#         fpath = "%s/%s" % (targets_dir, fname)
#         num_seq = count_lines(fpath, sep=bytes('>', 'utf8'))
#         fasta_src = parse_fasta(open(fpath, 'r'), 'fasta')
#         trg2seq.update(FastaFileLoader(fasta_src, num_seq).load())
#
#     return trg2seq


def load_training_and_validation(db, limit=None):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$lte": t0},
               'Aspect': ASPECT}

    sequences_train, annotations_train, _ = _get_labeled_data(db, q_train, None)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$gt": t0, "$lte": t1},
               'Aspect': ASPECT}

    sequences_valid, annotations_valid, _ = _get_labeled_data(db, q_valid, limit)
    forbidden = set(sequences_train.keys())
    sequences_valid = {k: v for k, v in sequences_valid.items() if k not in forbidden}
    annotations_valid = {k: v for k, v in annotations_valid.items() if k not in forbidden}

    return sequences_train, annotations_train, sequences_valid, annotations_valid


def _get_labeled_data(db, query, limit, propagate=True):

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    seqid2goid, goid2seqid = GoAnnotationCollectionLoader(s, c, ASPECT).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    if propagate:
        for k, v in seqid2goid.items():
            annots = ONTO.propagate(v, include_root=False)
            seqid2goid[k] = annots

    return seqid2seq, seqid2goid, goid2seqid


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
    blastdb_pth = os.path.join(tmp_dir, 'blast-%s' % GoAspect(ASPECT))
    if os.path.exists(blastdb_pth): return
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


def precision(tau, predictions, targets, classes=None):

    P, T = [], []
    for seqid, annotations in predictions.items():
        if classes:
            seq_annots = bin2dict(annotations, classes)
            seq_targets = bin2dict(targets[seqid], classes)
        else:
            seq_annots, seq_targets = annotations, targets[seqid]
        preds = [go for go, prob in seq_annots.items() if prob >= tau]
        if len(preds) == 0:
            continue
        P.append(set(preds))
        T.append(set(seq_targets))

    assert len(P) == len(T)

    if len(P) == 0: return 1.0

    total = sum([len(P_i & T_i) / len(P_i) for P_i, T_i in zip(P, T)])
    return total / len(P)


def recall(tau, predictions, targets, classes=None, partial_evaluation=False):

    P, T = [], []
    for seqid, annotations in predictions.items():
        if classes:
            annotations = bin2dict(annotations, classes)
            seq_targets = bin2dict(targets[seqid], classes)
        else:
            seq_annots, seq_targets = annotations, targets[seqid]
        preds = [go for go, prob in annotations.items() if prob >= tau]
        if not partial_evaluation and len(annotations) == 0: continue
        P.append(set(preds))
        T.append(set(seq_targets))

    assert len(P) == len(T)

    if len(P) == 0: return 0.0

    total = sum([len(P_i & T_i) / len(T_i) for P_i, T_i in zip(P, T)])
    return total / len(T)


def F_beta(pr, rc, beta=1):
    if rc == 0 and pr == 0:
        return np.nan
    return (1 + beta ** 2) * ((pr * rc) / (((beta ** 2) * pr) + rc))


def F_max(P, T, thresholds=THRESHOLDS):
    return np.max([F_beta(precision(th, P, T), recall(th, P, T)) for th in thresholds])


def predict(reference_seqs, reference_annots, target_seqs, method, load_file=True):
    if method == "blast":
        pred_path = os.path.join(tmp_dir, 'pred-blast-%s.npy' % GoAspect(ASPECT))
        if load_file and os.path.exists(pred_path):
            return np.load(pred_path).item()
        _prepare_blast(reference_seqs)
        predictions = _predict(reference_annots, target_seqs, _blast)
        np.save(pred_path, predictions)
        return predictions
    elif method == "naive":
        _prepare_naive(reference_annots)
        predictions = _predict(reference_annots, target_seqs, _naive)
        return predictions
    elif method == "seq2go":
        pred_path = os.path.join(tmp_dir, 'pred-seq2go-%s.npy' % GoAspect(ASPECT))
        return np.load(pred_path).item()
    elif method == "seq2go-proba":
        pred_path = os.path.join(tmp_dir, 'pred-seq2go-proba-%s.npy' % GoAspect(ASPECT))
        return np.load(pred_path).item()
    else:
        print("Unknown method")


def performance(predictions, ground_truth, thresholds=THRESHOLDS):
    P, T = predictions, ground_truth
    prs = [precision(th, P, T) for th in thresholds]
    rcs = [recall(th, P, T) for th in thresholds]
    f1s = [F_beta(pr, rc) for pr, rc in zip(prs, rcs)]
    return prs, rcs, f1s


def plot_precision_recall(perf):
    # Plot Precision-Recall curve
    lw, n = 2, len(perf)
    methods = list(perf.keys())
    prs = [v[0] for v in perf.values()]
    rcs = [v[1] for v in perf.values()]
    f1s = [v[2] for v in perf.values()]

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


def evaluate_performance(db, methods, asp):
    init_GO(asp)
    lim = None
    seqs_train, annots_train, seqs_valid, annots_valid = \
        load_training_and_validation(db, lim)
    perf = {}
    for meth in methods:
        pred = predict(seqs_train, annots_train, seqs_valid, meth)
        perf[meth] = performance(pred, annots_valid)
    plot_precision_recall(perf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # load_evaluation_data()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    lim = 100
    init_GO(ASPECT)

    seqs_train, annots_train, seqs_valid, annots_valid = load_training_and_validation(db, lim)
    y_blast = predict(seqs_train, annots_train, seqs_valid, "blast")
    performance(y_blast, annots_valid)
