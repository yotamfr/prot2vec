import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from pymongo import MongoClient

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

import utils
logger = utils.get_logger("baselines")

# from src.interactions import STRINGSPECEIS
# from src.uniprot_pfam_goa_mongo import exp_codes

from prot2vec import Node2Vec
from ecod import EcodDomain

client = MongoClient('mongodb://localhost:27017/')
db = client['prot2vec']

import parameters as params

args = params.arguments

ckptpath = args["ckpt_path"]
seq_length = args["seq_length"]
emb_dim = args["word_embedding_dim"]
datapath = args["data_path"]

ONECLASSDIR = "%s/oneclass" % datapath
TARGETSEQDIR = "../CAFA/CAFA3/CAFA3_targets/Target files"
TRAININGDATA = "../CAFA/CAFA3/CAFA3_training_data/uniprot_sprot_exp.fasta"
TRAININGLABELS = "../CAFA/CAFA3/CAFA3_training_data/uniprot_sprot_exp.txt"
HMMERBATCHSIZE = 10

# CAFA2_cutoff = datetime.datetime(2013, 6, 1, 0, 0)
CAFA2_cutoff = datetime.datetime(2014, 1, 1, 0, 0)
now = datetime.datetime.utcnow()
# now = CAFA2_cutoff

WORD2VEC = Node2Vec()

ONECLASS = {
    "IsolationForest": IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0,
                                       bootstrap=False, n_jobs=1, random_state=None, verbose=0),
    "OneClassSVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
    "EllipticEnvelope": EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=None,
                         contamination=0.1, random_state=None)
}

BINARY = {
    "SVM": SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
               coef0=0.0, shrinking=True, probability=False, tol=0.001,
               cache_size=200, class_weight=None, verbose=False,
               max_iter=-1, decision_function_shape=None, random_state=None),
    "RFC": RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None,
                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                               max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07,
                               bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                               warm_start=False, class_weight=None)
}


def load_ecod_data(sample_size):

    ids, wordvecs, annots, terms = [], [], [], set()

    # sequences = map(EcodDomain, db.ecod.aggregate([{"$sample": {"size": sample_size}}]))

    sequences = map(EcodDomain, db.ecod.find({}))

    # for _ in tqdm(range(sample_size), desc="ECOD sequences processed"):
    for _ in tqdm(range(db.ecod.count({})), desc="ECOD sequences processed"):
        seq = next(sequences)
        # key = seq.eid.lower()
        # if key not in WORD2VEC:
        #     continue
        key = seq.eid
        if key not in WORD2VEC.model:
            continue
        goterms = seq.get_go_terms()
        if not len(goterms):
            continue
        wordvecs.append(WORD2VEC[key])
        annots.append(goterms)
        terms |= goterms
        ids.append(key)

    X = np.array(wordvecs)
    mlb = MultiLabelBinarizer(classes=list(terms), sparse_output=False)
    y = mlb.fit_transform(annots)

    logger.info("X.shape=%s y.shape=%s\n" % (X.shape, y.shape))

    return X, y, list(mlb.classes_)


def train_multicalss_classifiers(X_train, y_train, X_test, y_test, datadir=ONECLASSDIR, calssifier=""):
    pass


def train_oneclass_classifiers(X_train, y_train, calssifier="OneClassSVM"):
    logger.info("Training Oneclass Classifiers ...\n")
    bar = tqdm(range(y_train.shape[1]), desc="classes trained")
    for j in bar:
        clf = ONECLASS[calssifier]
        ix = y_train[:, j]
        obs = X_train[np.logical_and(ix, np.ones(ix.shape))]
        if not len(obs):
            continue
        clf.fit(obs)
        joblib.dump(clf, '%s/%s.%s.pkl' % (ONECLASSDIR, calssifier, j))


def train_multilablel_classifiers(X, y):
    logger.info("Training multi-lablel MLP ...\n")
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    clf.fit(X, y)
    joblib.dump(clf, '%s/MLPClassifier.pkl' % (ONECLASSDIR,))


def measure_oneclass_accuracy(X_test, y_truth, calssifier="OneClassSVM"):
    logger.info("Measuring accuracy of %s Classifiers ...\n" % calssifier)
    y_preds = np.zeros(y_truth.shape)               # all zeros matrix
    score = (0., 0., 0., 0, 0)
    optf1 = 0.0
    tmpl = "score --> precision=%s, recall=%s, f1_score=%s, #truths=%s, #labels=%s"
    bar = tqdm(range(y_truth.shape[1]), desc=tmpl % score)
    for j in bar:
        model_filname = '%s/%s.%s.pkl' % (ONECLASSDIR, calssifier, j)
        if not os.path.exists(model_filname):
            continue
        clf = joblib.load(model_filname)
        y_preds[:, j] = np.max([clf.predict(X_test), y_preds[:, j]], axis=0)
        score = precision_recall_fscore_support(y_truth[:, j], y_preds[:, j], beta=1.0, average="binary")
        if score[2] > optf1:
            bar.set_description(tmpl % (score[0], score[1], score[2], np.sum(y_truth[:, j]), np.sum(y_preds[:, j])))
            optf1 = score[2]
    score = precision_recall_fscore_support(y_truth, y_preds, beta=1.0, average="micro")
    logger.info(tmpl % (score[0], score[1], score[2], np.sum(y_truth), np.sum(y_preds)))
    return score


def get_hmmer_predictions(datafile=TRAININGDATA, date=now):

    logger.info("Predicting labels for %s based on HMMER.\n" % datafile)

    numseqs, intuitive_prediction = 0, {}
    fasta_sequences = SeqIO.parse(open(datafile, 'rU'), 'fasta')
    for target in fasta_sequences: numseqs += 1
    fasta_sequences = SeqIO.parse(open(datafile, 'rU'), 'fasta')

    for i in tqdm(range(numseqs)):
        target = next(fasta_sequences)
        hits = sorted(Hmmer.find_one({"_id": target.id})["hits"],
                      key=lambda h: h['@score'], reverse=True)
        intuitive_prediction[target.id] = set()
        for hit in hits:
            terms = get_goa_until_date(hit["@acc2"], date)
            if not(hit["@acc2"] == target.id or terms.issubset(set())):
                intuitive_prediction[target.id] = terms
                break
    return intuitive_prediction


def get_goa_until_date(id, date):
    annots = Annots.find({"DB_Object_ID": id, "Date": {"$lte": date}, "ECO_Evidence_code": {"$in": exp_codes}})
    return set(map(lambda annot: annot["GO_ID"], annots))


def get_mock_predictions(datafile=TRAININGDATA, date=now):

    logger.info("Predicting labels for %s based on GOA.\n" % datafile)

    numseqs, intuitive_prediction = 0, {}
    fasta_sequences = SeqIO.parse(open(datafile, 'rU'), 'fasta')
    for target in fasta_sequences: numseqs += 1
    fasta_sequences = SeqIO.parse(open(datafile, 'rU'), 'fasta')

    for i in tqdm(range(numseqs)):
        target = next(fasta_sequences)
        terms = get_goa_until_date(target.id, date)
        intuitive_prediction[target.id] = set(terms)
    return intuitive_prediction


def compute_precision_recall_fscore_support(intuitive_truths, intuitive_predictions, limit=10000):

    st = set(intuitive_truths.keys())
    sp = set(intuitive_predictions.keys())

    assert set(st).issubset(sp) and len(sp) == len(st)

    st = list(st)[:limit]
    targets, truths, preds, cls = [], [], [], set()
    for i in tqdm(range(len(st)), desc="targets processed"):
        key = st[i]
        targets.append(key)
        truths.append(intuitive_truths[key])
        preds.append(intuitive_predictions[key])
        cls |= intuitive_predictions[key]
        cls |= intuitive_truths[key]

    mlb = MultiLabelBinarizer(classes=list(cls))
    y_true = mlb.fit_transform(truths)
    y_pred = mlb.fit_transform(preds)

    logger.info("Found %s different GeneOntology Terms" % y_true.shape[1])

    return precision_recall_fscore_support(y_true, y_pred, beta=1.0, average="micro")


def rank_predictions(intuitive_truths, intuitive_predictions, mode=2):

    logger.info("Computing Precision-Recall-Fscore-Support.\n")

    if mode == 1:

        score = compute_precision_recall_fscore_support(intuitive_truths, intuitive_predictions)

        logger.info("precision=%s, recall=%s, f1_score=%s, support=%s" % score)

    elif mode == 2:

        truths, predictions = remove_null_preds(intuitive_truths, intuitive_predictions)

        score = compute_precision_recall_fscore_support(truths, predictions)

        logger.info("precision=%s, recall=%s, f1_score=%s, support=%s" % score)

    else:
        logger.error("Unknownwn assessment mode")


def remove_null_preds(intuitive_truths, intuitive_predictions):

    null_keys = list(filter(lambda key: intuitive_predictions[key].issubset(set())
                       , intuitive_predictions.keys()))
    for key in null_keys:
        try:
            del intuitive_predictions[key]
            del intuitive_truths[key]
        except KeyError as err:
            logger.error(err)

    return intuitive_truths, intuitive_predictions


def load_word2vec_models():
    WORD2VEC.load("%s/ecod.dense.emb" % ckptpath)


if __name__ == "__main__":

    load_word2vec_models()

    data, labels, classes = load_ecod_data(100000)
    macro, micro, support, num_classes = 0.0, 0.0, 0, 0

    logger.info("Training Classifiers ...\n")

    for j in range(labels.shape[1]):

        X, y = data, labels[:, j]

        kfold = 5
        sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.2, random_state=0)

        try:
            scores = np.zeros(3)
            tmpl = "{0}: precision={1:.2f} recall={2:.2f} f1_score={3:.2f} support={4:4d}"

            for train_index, test_index in sss.split(X, y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_valid = y[train_index], y[test_index]

                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                clf = BINARY["SVM"]
                clf.fit(X_train, y_train)

                y_preds = np.zeros(y_valid.shape)  # all zeros matrix
                y_preds = np.max([clf.predict(X_test), y_preds], axis=0)

                pr, rc, f1, _ = precision_recall_fscore_support(y_valid, y_preds, beta=1.0, average="binary")
                scores += (pr, rc, f1)

            pr, rc, f1 = scores / kfold
            num_classes += 1
            support += sum(y)
            macro += f1
            micro += f1*sum(y)

            logger.info(tmpl.format(classes[j], pr, rc, f1, sum(y)))

        except ValueError as err:
            logger.error(err)

    logger.info("f1_macro: %s, f1_micro: %s, total_support: %s" %
                (macro/num_classes, micro/support, support))
