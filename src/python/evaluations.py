import os
import datetime
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient

from Bio import SeqIO

import matplotlib.pyplot as plt

from scipy import interp

from itertools import cycle

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import utils
logger = utils.get_logger("evaluations")



from prot2vec import Node2Vec, Clstr2Vec, BagOfNGrams

from models import PdbChain
from models import Uniprot

client = MongoClient('mongodb://localhost:27017/')
db = client['prot2vec']

import parameters as params

args = params.arguments

ckptpath = args["ckpt_path"]
seq_length = args["seq_length"]
emb_dim = args["word_embedding_dim"]
datapath = args["data_path"]

ONECLASSDIR = "%s/../oneclass" % datapath
TARGETSEQDIR = "../CAFA/CAFA3/CAFA3_targets/Target files"
TRAININGDATA = "../CAFA/CAFA3/CAFA3_training_data/uniprot_sprot_exp.fasta"
TRAININGLABELS = "../CAFA/CAFA3/CAFA3_training_data/uniprot_sprot_exp.txt"
HMMERBATCHSIZE = 10

# CAFA2_cutoff = datetime.datetime(2013, 6, 1, 0, 0)
CAFA2_cutoff = datetime.datetime(2014, 1, 1, 0, 0)
CAFA3_cutoff = datetime.datetime(2017, 1, 1, 0, 0)
now = datetime.datetime.utcnow()

random_state = np.random.RandomState(0)
np.random.seed(101)

WORD2VEC = Node2Vec()

ONECLASS = {
    "IsolationForest": IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0,
                                       bootstrap=False, n_jobs=1, random_state=random_state, verbose=0),
    "OneClassSVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1),
    "EllipticEnvelope": EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=None,
                         contamination=0.1, random_state=random_state)
}

BINARY = {
    "SVM": SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
               coef0=0.0, shrinking=True, probability=False, tol=0.001,
               cache_size=200, class_weight=None, verbose=False,
               max_iter=-1, decision_function_shape=None, random_state=random_state),
    "RFC": RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None,
                                  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                  max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07,
                                  bootstrap=True, oob_score=False, n_jobs=1, random_state=random_state, verbose=0,
                                  warm_start=False, class_weight=None)
}


def load_validation_set(annots_tsv, seqs_fasta, aspect=None):

    logger.info("Loading validation set.")

    validation_set, validation_seqs, validation_terms = dict(), dict(), dict()

    fasta_sequences = SeqIO.parse(open(seqs_fasta), 'fasta')
    for fasta in fasta_sequences:
        validation_seqs[fasta.id] = fasta.seq

    with open(annots_tsv, 'r') as f:
        for line in f:
            seq_id, go_id, go_asp = line.strip().split('\t')
            if aspect and go_asp != aspect:
                continue
            try:
                if go_id in validation_terms:
                    validation_terms[go_id] += 1
                else:
                    validation_terms[go_id] = 1
                if seq_id in validation_set:
                    validation_set[seq_id].add(go_id)
                else:
                    validation_set[seq_id] = {go_id}
            except TypeError:
                pass

    logger.info("Loaded %s annots." % sum(validation_terms.values()))

    return validation_set, validation_seqs, validation_terms


def load_training_set(go_terms, collection, Model):

    logger.info("Loading training set.")

    query = {"DB": "UniProtKB",
             "GO_ID": {"$in": list(go_terms)},
             "Date": {"$lte": CAFA3_cutoff}}
    training_annots = db.goa_uniprot.find(query)
    training_num = db.goa_uniprot.count(query)

    training_set, training_seqs, training_terms = dict(), dict(), dict()

    for _ in tqdm(range(training_num), desc="sequences processed"):
        go_doc = next(training_annots)
        seq_id = go_doc["DB_Object_ID"]
        go_id = go_doc["GO_ID"]
        seq_doc = collection.find_one({"_id": seq_id})
        if not seq_doc:
            continue
        seq = Model(seq_doc)
        if seq_id not in training_seqs:
            training_seqs[seq_id] = seq.seq
        if go_id in training_terms:
            training_terms[go_id] += 1
        else:
            training_terms[go_id] = 1
        if seq_id in training_set:
            training_set[seq_id].add(go_id)
        else:
            training_set[seq_id] = {go_id}

    logger.info("Loaded %s annots." % sum(training_terms.values()))

    return training_set, training_seqs, training_terms


def validate_sequence(seq_id, seq, collection, Model, models):
    doc = collection.find_one({"_id": seq_id})
    if not doc:
        return False
    prot = Model(doc)
    if prot.seq != seq:
        return False
    if not np.all([seq_id in D for D in models]):
        return False
    return True


def load_multiple_data2(annots_tsv, seqs_fasta, collection, Model, models,
                        aspect=None, plot_hist=True):

    validation_set, validation_seqs, go_terms = load_validation_set(annots_tsv, seqs_fasta, aspect)

    mlb = MultiLabelBinarizer(classes=list(go_terms.keys()), sparse_output=True)

    valid_ids, valid_X, valid_y = [], {D.name: [] for D in models}, []

    pbar1 = tqdm(range(len(validation_set)), desc="sequences processed")
    for seq_id, annots in validation_set.items():
        pbar1.update(1)
        if not validate_sequence(seq_id, validation_seqs[seq_id], collection, Model, models):
            continue
        valid_ids.append(seq_id)
        valid_y.append(annots)
    pbar1.close()

    for D in models:
        valid_X[D.name] = D.wordvecs(valid_ids)

    valid_y = mlb.fit_transform(valid_y)

    logger.info("valid_X.shape=%s valid_y.shape=%s\n"
                % (valid_X[models[0].name].shape, valid_y.shape))

    if plot_hist:
        plt.hist(list(map(lambda annots: len(annots), validation_set.values())), bins=200)
        plt.title("Annotations per sequence")
        plt.xlabel("#GO-terms annotations")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        plt.hist(list(filter(lambda val: val < 200, go_terms.values())), bins=200)
        plt.title("Annotations per GO-term")
        plt.xlabel("#Annotations in data")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    training_set, training_seqs, _ = load_training_set(go_terms.keys(), collection, Model)

    train_ids, train_X, train_y = [], {D.name: [] for D in models}, []

    pbar2 = tqdm(range(len(training_set)), desc="sequences processed")
    for seq_id, annots in training_set.items():
        pbar2.update(1)
        if not np.all([seq_id in D for D in models]):
            continue
        train_ids.append(seq_id)
        train_y.append(annots)
    pbar2.close()

    for D in models:
        train_X[D.name] = D.wordvecs(train_ids)

    train_y = mlb.transform(train_y)

    logger.info("train_X.shape=%s train_y.shape=%s\n"
                % (train_X[models[0].name].shape, train_y.shape))

    return train_X, valid_X, train_y, valid_y, mlb.classes_


def load_data(sample_size, collection, Model, aspect=None):

    ids, wordvecs, annots, terms = [], [], [], set()

    sample = collection.aggregate([{"$sample": {"size": sample_size}}])

    for _ in tqdm(range(sample_size), desc="sequences processed"):
        seq = Model(next(sample))
        if not seq.is_gene_product():
            continue
        key = seq.name
        if key not in WORD2VEC:
            continue
        goterms = seq.get_go_terms(aspect)
        if not len(goterms):
            continue
        wordvecs.append(WORD2VEC[key])
        annots.append(goterms)
        terms |= goterms
        ids.append(key)

    X = np.matrix(wordvecs)
    mlb = MultiLabelBinarizer(classes=list(terms), sparse_output=False)
    y = mlb.fit_transform(annots)

    logger.info("X.shape=%s y.shape=%s\n" % (X.shape, y.shape))

    return X, y, mlb.classes_


def load_multiple_data(sample_size, collection, Model, dictionaries=[WORD2VEC], aspect=None):

    sample = collection.aggregate([{"$sample": {"size": sample_size}}])

    words, wordvecs, annots, terms = [], {D.name: [] for D in dictionaries}, [], set()

    for _ in tqdm(range(sample_size), desc="sequences processed"):
        seq = Model(next(sample))
        if not seq.is_gene_product():
            continue
        key = seq.name
        if not np.all([key in D for D in dictionaries]):
            continue
        goterms = seq.get_go_terms(aspect)
        if not len(goterms):
            continue
        annots.append(goterms)
        terms |= goterms
        words.append(key)

    for D in dictionaries:
        wordvecs[D.name] = D.wordvecs(words)

    mlb = MultiLabelBinarizer(classes=list(terms), sparse_output=False)
    y = mlb.fit_transform(annots)

    logger.info("X.shape=%s y.shape=%s\n" % (wordvecs[dictionaries[0].name].shape, y.shape))

    return wordvecs, y, mlb.classes_


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


# def get_goa_until_date(id, date):
#     annots = Annots.find({"DB_Object_ID": id, "Date": {"$lte": date}, "ECO_Evidence_code": {"$in": exp_codes}})
#     return set(map(lambda annot: annot["GO_ID"], annots))


# def get_mock_predictions(datafile=TRAININGDATA, date=now):
#
#     logger.info("Predicting labels for %s based on GOA.\n" % datafile)
#
#     numseqs, intuitive_prediction = 0, {}
#     fasta_sequences = SeqIO.parse(open(datafile, 'rU'), 'fasta')
#     for target in fasta_sequences: numseqs += 1
#     fasta_sequences = SeqIO.parse(open(datafile, 'rU'), 'fasta')
#
#     for i in tqdm(range(numseqs)):
#         target = next(fasta_sequences)
#         terms = get_goa_until_date(target.id, date)
#         intuitive_prediction[target.id] = set(terms)
#     return intuitive_prediction


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
    # WORD2VEC.load("%s/ecod.dense.emb" % ckptpath)
    # WORD2VEC.load("%s/ecod.simple.emb" % ckptpath)
    # WORD2VEC.load("%s/pdb.60.emb" % ckptpath)
    WORD2VEC.load("%s/uniprot.60.emb" % ckptpath)
    # WORD2VEC.load("%s/random.emb" % ckptpath)


def main1(models, collection, Model, sample_size):

    for model in models:
        WORD2VEC.load("%s/%s.emb" % (ckptpath, model))

    data, labels, classes = load_data(sample_size, collection, Model)
    macro, micro, support, num_classes = 0.0, 0.0, 0, 0

    logger.info("Training Classifiers ...\n")

    for j in range(labels.shape[1]):

        X, y = data, labels[:, j]

        kfold = 5

        if sum(y) < 10: continue

        sss = StratifiedShuffleSplit(n_splits=kfold, test_size=.2, random_state=0)

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


def main2(models, collection, Model, sample_size):

    for model in models:
        WORD2VEC.load("%s/%s.emb" % (ckptpath, model))

    data, labels, classes = load_data(sample_size, collection, Model)

    X, y = data, labels

    logger.info("filtering...")

    ix1 = np.sum(y, axis=0) > 9
    y = y[:, ix1]
    classes = classes[ix1]
    ix0 = np.sum(y, axis=1) > 0
    y = y[ix0, :]
    X = X[ix0, :]

    logger.info("X.shape=%s y.shape=%s\n" % (X.shape, y.shape))

    logger.info("training...")

    n_classes = y.shape[1]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                            random_state=0, stratify=y)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(BINARY["SVM"], n_jobs=2)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[classes[i]], tpr[classes[i]], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[classes[i]] = auc(fpr[classes[i]], tpr[classes[i]])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[classes[i]] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[classes[i]], tpr[classes[i]])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        title = '#Sequences: %s, #GO-Terms: %s' % y.shape
        plot_roc_auc(fpr, tpr, roc_auc, classes, title)

    except ValueError as err:
        logger.error(err)


def plot_roc_auc(fpr, tpr, roc_auc, classes, title):
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta', 'cyan'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[classes[i]], tpr[classes[i]], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[classes[i]]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def main3(dictionaries, collection, Model, sample_size):

    for model in dictionaries:
        model.load("%s/%s" % (ckptpath, model.name))

    wordsvecs, labels, classes = load_multiple_data(sample_size, collection, Model, dictionaries)

    for model in dictionaries:

        logger.info("Measuring %s Performance" % model.name)

        X = wordsvecs[model.name]

        logger.info("filtering...")

        ix1 = np.sum(labels, axis=0) > 9       # only cls when num samples > 9
        y = labels[:, ix1]
        cls = classes[ix1]
        ix0 = np.sum(y, axis=1) > 0       # only samples when mum cls > 0
        y = y[ix0, :]
        X = X[ix0, :]

        assert np.all(np.sum(y, axis=0) > 9)
        assert np.all(np.sum(y, axis=1) > 0)

        logger.info("X.shape=%s y.shape=%s\n" % (X.shape, y.shape))

        logger.info("training...")

        n_classes = y.shape[1]

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Learn to predict each class against the other
            classifier = OneVsRestClassifier(BINARY["SVM"], n_jobs=4)
            y_score = classifier.fit(X_train, y_train).decision_function(X_test)

            # Compute ROC curve and ROC area for each class
            precision = dict()
            recall = dict()
            average_precision = dict()
            for i in range(n_classes):
                precision[cls[i]], recall[cls[i]], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
                average_precision[cls[i]] = average_precision_score(y_test[:, i], y_score[:, i])

                # Compute Precision-Recall and plot curve
                precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
                average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

            # Compute micro-average ROC curve and ROC area
            precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
            average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

            title = 'Precision-Recall %s: #Sequences: %s, #GO-Terms: %s' % (model.name, y.shape[0], y.shape[1])
            plot_precision_recall(recall, precision, average_precision, cls, title)

        except ValueError as err:
            logger.error(err)


def plot_precision_recall(recall, precision, average_precision, classes, title):
    # Plot Precision-Recall curve
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw, n_classes = 2, 5

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[classes[i]], precision[classes[i]], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], average_precision[classes[i]]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def main4(models, aspect, sample_size):

    collection, Model = db.pdb, PdbChain

    for model in models:
        WORD2VEC.load("%s/%s.emb" % (ckptpath, model))

    data, labels, classes = load_data(sample_size, collection, Model, aspect)

    X, y = data, labels

    logger.info("filtering...")

    ix1 = np.sum(y, axis=0) > 9
    y = y[:, ix1]
    classes = classes[ix1]
    ix0 = np.sum(y, axis=1) > 0
    y = y[ix0, :]
    X = X[ix0, :]

    logger.info("X.shape=%s y.shape=%s (%s)\n" % (X.shape, y.shape, aspect))

    logger.info("training...")

    n_classes = y.shape[1]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                            random_state=0, stratify=y)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(BINARY["SVM"], n_jobs=2)
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)

        # Compute ROC curve and ROC area for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[classes[i]], recall[classes[i]], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            average_precision[classes[i]] = average_precision_score(y_test[:, i], y_score[:, i])

            # Compute Precision-Recall and plot curve
            precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
            average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

        title = 'Precision-Recall curve: #Sequences: %s, #GO-Terms: %s' % y.shape
        plot_precision_recall(recall, precision, average_precision, classes, title)

    except ValueError as err:
        logger.error(err)

if __name__ == "__main__":

    load_multiple_data2(args["cafa3_sprot_goa"], args["cafa3_sprot_seq"],
                        db.uniprot, Uniprot, [BagOfNGrams(3)])

    main3([Node2Vec("random.uniprot.emb"),
           Node2Vec("intact.all.emb"),
           Node2Vec("uniprot.60.emb"),
           # Node2Vec("uniprot.80.emb"),
           # Clstr2Vec("sprot.clstr.60.json")
           ], db.uniprot, Uniprot, 50000)

    main3([Node2Vec("random.pdb.emb"),
           Node2Vec("pdbnr.complex.emb"),
           Node2Vec("pdbnr.enriched.emb")
           ], db.pdbnr, PdbChain, 50000)

    # main4(["pdb.60"], "F", 50000)
