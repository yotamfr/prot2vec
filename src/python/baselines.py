import os

from .preprocess import *

import subprocess

from tqdm import tqdm

from Bio.Seq import Seq
from Bio import SeqIO, SearchIO
from Bio.SeqRecord import SeqRecord

from Bio.Blast.Applications import NcbiblastpCommandline

import argparse

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()


CUTOFF_DATE = cafa3_cutoff
ASPECT = 'F'
ONTO = None
PRIOR = {}


def init_GO(asp=ASPECT):
    global ONTO, ASPECT
    ASPECT = asp
    ONTO = get_ontology(asp)


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")


def load_all_data():
    mf, _ = load_data(db, asp='F', codes=exp_codes)
    cc, _ = load_data(db, asp='C', codes=exp_codes)
    bp, _ = load_data(db, asp='P', codes=exp_codes)
    return mf, cc, bp


def get_training_and_validation(db, limit=None):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$lte": CUTOFF_DATE},
               'Aspect': ASPECT}

    sequences_train, annotations_train, _ = _get_labeled_data(db, q_train, None)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$gt": CUTOFF_DATE},
               'Aspect': ASPECT}

    sequences_valid, annotations_valid, _ = _get_labeled_data(db, q_valid, limit)
    forbidden = set(sequences_train.keys())
    sequences_valid = {k: v for k, v in sequences_valid.items() if k not in forbidden}
    annotations_valid = {k: v for k, v in annotations_valid.items() if k not in forbidden}

    return sequences_train, annotations_train, sequences_valid, annotations_valid


def _get_labeled_data(db, query, limit):

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    seqid2goid, goid2seqid = GoAnnotationCollectionLoader(s, c, ASPECT).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    for k, v in seqid2goid.items():
        seqid2goid[k] = ONTO.augment(v)

    return seqid2seq, seqid2goid, goid2seqid


def _prepare_naive(reference):
    global PRIOR
    for _, go_terms in reference.items():
        for go in go_terms:
            if go in PRIOR:
                PRIOR[go] += 1
            else:
                PRIOR[go] = 1
    total = sum(PRIOR.values())
    return {go: count/total for go, count in PRIOR.items()}


def _naive(target, reference):
    return PRIOR


def _prepare_blast2go(sequences):
    records = [SeqRecord(Seq(seq), id) for id, seq in sequences.items()]
    blastdb_pth = os.path.join(tmp_dir, 'blast2go-%s' % GoAspect(ASPECT))
    SeqIO.write(records, open(blastdb_pth, 'w+'), "fasta")
    os.system("makeblastdb -in %s -dbtype prot" % blastdb_pth)


def _blast2go(target, reference, topn=None, choose_max_prob=True):

    query_pth = os.path.join(tmp_dir, 'query-%s.fasta' % GoAspect(ASPECT))
    output_pth = os.path.join(tmp_dir, "blastp-%s.out" % GoAspect(ASPECT))
    database_pth = os.path.join(tmp_dir, 'blast2go-%s' % GoAspect(ASPECT))

    SeqIO.write(SeqRecord(Seq(target), "QUERY"), open(query_pth, 'w+'), "fasta")

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
    return annotations


def _predict(reference_annots, target_seqs, func_predict):

    pbar = tqdm(range(len(target_seqs)), desc="targets processed")

    y_pred = np.zeros((len(target_seqs), len(ONTO.classes)))
    for i, (id, seq) in enumerate(target_seqs.items()):
        preds = func_predict(seq, reference_annots)
        bin_preds = ONTO.binarize([list(preds.keys())])[0]
        for go, prob in preds.items():
            bin_preds[ONTO[go]] = prob
        y_pred[i, :] = bin_preds
        pbar.update(1)
    pbar.close()

    return y_pred


def precision(tau, P, T):
    P = np.where(P >= tau, 1, 0)
    ix = np.array(list(map(lambda row: np.sum(row) > 0, P)))
    P, T = P[ix, :], T[ix, :]
    m_th, _ = P.shape   # m(tau)
    intersection = np.where(P + T == 2, 1, 0)
    total = np.sum([np.sum(intersection[i, :]) / np.sum(P[i, :]) for i in range(m_th)])
    return total / m_th


def recall(tau, P, T, partial_evaluation=False):
    if partial_evaluation:
        ix = np.array(list(map(lambda row: np.sum(row) > 0, P)))
        P, T = P[ix, :], T[ix, :]        # n_e = m(0)
    n_e, _ = T.shape    # n_e = n
    P = np.where(P >= tau, 1, 0)
    intersection = np.where(P + T == 2, 1, 0)
    total = np.sum([np.sum(intersection[i, :]) / np.sum(T[i, :]) for i in range(n_e)])
    return total / n_e


def F_beta(pr, rc, beta=1):
    return (1 + beta ** 2) * ((pr * rc) / (((beta ** 2) * pr) + rc))


def F_max(P, T, thresholds=np.arange(0.01, 1, 0.1)):
    return np.max([F_beta(precision(th, P, T), recall(th, P, T)) for th in thresholds])


def predict(reference_seqs, reference_annots, target_seqs, method="blast2go"):
    pred_path = os.path.join(tmp_dir, 'pred-%s-%s.npy' % (method, GoAspect(ASPECT)))
    if os.path.exists(pred_path):
        return np.load(pred_path)
    if method == "blast2go":
        _prepare_blast2go(reference_seqs)
        y_pred = _predict(reference_annots, target_seqs, _blast2go)
        np.save(pred_path, y_pred)
        return y_pred
    elif method == "naive":
        _prepare_naive(reference_seqs)
        y_pred = _predict(reference_annots, target_seqs, _naive)
        np.save(pred_path, y_pred)
        return y_pred
    else:
        print("Unknown method")


def evaluate(y_pred, ground_truth):
    y_truth = ONTO.binarize([v for k, v in ground_truth.items()])
    print(F_max(y_pred, y_truth))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    lim = 100
    init_GO(ASPECT)

    seqs_train, annots_train, seqs_valid, annots_valid = get_training_and_validation(db, lim)
    y_blast2go = predict(seqs_train, annots_train, seqs_valid, "blast2go")
    evaluate(y_blast2go, annots_valid)

