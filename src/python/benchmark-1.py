import os

from src.python.consts import *

from src.python.preprocess import *

from src.python.geneontology import *

from pymongo import MongoClient


### Keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Concatenate, Flatten


def _get_labeled_data(db, query, limit, pssm=True):

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    seqid2goid, _ = GoAnnotationCollectionLoader(s, c, ASPECT).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}

    if pssm:
        num_seq = db.pssm.count(query)
        src_seq = db.pssm.find(query)
        seqid2seq = PssmCollectionLoader(src_seq, num_seq).load()
    else:
        num_seq = db.uniprot.count(query)
        src_seq = db.uniprot.find(query)
        seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    seqid2goid = {k: v for k, v in seqid2goid.items() if len(v) > 1 or 'GO:0005515' not in v}
    seqid2goid = {k: v for k, v in seqid2goid.items() if k in seqid2seq.keys()}

    return seqid2seq, seqid2goid


def load_training_and_validation(db, limit=None):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$lte": t0},
               'Aspect': ASPECT}

    sequences_train, annotations_train = _get_labeled_data(db, q_train, limit)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$gt": t0, "$lte": t1},
               'Aspect': ASPECT}

    sequences_valid, annotations_valid = _get_labeled_data(db, q_valid, limit)
    forbidden = set(sequences_train.keys())
    sequences_valid = {k: v for k, v in sequences_valid.items() if k not in forbidden}
    annotations_valid = {k: v for k, v in annotations_valid.items() if k not in forbidden}

    return sequences_train, annotations_train, sequences_valid, annotations_valid


def data_generator(seq2pssm, seq2go, classes):
    n = len(classes)
    s_cls = set(classes)
    for i, (k, v) in enumerate(seq2go.items()):

        sys.stdout.write("\r{0:.0f}%".format(100.0 * i / len(seq2go)))

        if k not in seq2pssm:
            continue
        y = np.zeros(len(classes))
        for go in onto.propagate(seq2go[k]):
            if go not in s_cls:
                continue
            y[classes.index(go)] = 1

        seq, pssm, msa = seq2pssm[k]
        x1 = [[AA.aa2onehot[aa], [pssm[i][AA.index2aa[k]] for k in range(20)]] for i, aa in enumerate(seq)]
        x2 = msa

        yield k, x1, x2, y


def Motifs(inpt):
    initial = Conv2D(192, (1, 20), padding='same', activation='relu')(inpt)
    motif03 = Conv2D(64, (3, 1), padding='same', activation='relu')(initial)
    motif09 = Conv2D(64, (9, 1), padding='same', activation='relu')(initial)
    motif36 = Conv2D(64, (36, 1), padding='same', activation='relu')(initial)

    return Concatenate(axis=1)([motif03, motif09, motif36])


def Features(inpt):
    out = inpt
    out = Conv2D(64, (3, 1), activation='relu', padding='same')(out)
    out = Conv2D(64, (3, 1), activation='relu', padding='same')(out)
    out = MaxPooling2D((2, 1))(out)
    out = Conv2D(128, (3, 1), activation='relu', padding='same')(out)
    out = Conv2D(128, (3, 1), activation='relu', padding='same')(out)
    out = MaxPooling2D((2, 1))(out)

    out = GlobalMaxPooling2D(data_format='channels_first')(out)
    # return Flatten()(out)
    return out


def Classifier(inpt, hidden_size, classes):
    x = inpt
    # We stack a deep densely-connected network on top
    x = Dense(hidden_size * 2, activation='relu')(x)
    x = Dense(hidden_size, activation='relu')(x)

    # And finally we add the main logistic regression layer
    return Dense(len(classes), activation='tanh')(x)


def ModelCNN(classes):
    inp = Input(shape=(2, None, 20))
    out = Classifier(Features(Motifs(inp)), 64, classes)
    model = Model(inputs=[inp], outputs=[out])
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='hinge', optimizer=sgd)

    return model


def get_xy(gen, normalize=False):
    xByShapes, yByShapes = dict(), dict()
    for _, x, _, y in gen:
        x = np.array(x).reshape(2, len(x), 20)
        y = np.add(np.multiply(np.array(y), 2), -1)
        if normalize: x = np.divide(np.add(x, -np.mean(x)), np.std(x))
        if x.shape in xByShapes:
            xByShapes[x.shape].append(x)
            yByShapes[x.shape].append(y)
        else:
            xByShapes[x.shape] = [x]  # initially a list, because we're going to append items
            yByShapes[x.shape] = [y]

    xByShapes = {k: np.array(v) for k, v in xByShapes.items()}
    yByShapes = {k: np.array(v) for k, v in yByShapes.items()}

    return xByShapes, yByShapes


if __name__=="__main__":

    ASPECT = 'F'  # Molecular Function

    client = MongoClient("mongodb://127.0.0.1:27017")

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(ASPECT)

    trn_seq2pssm, trn_seq2go, tst_seq2pssm, tst_seq2go = \
        load_training_and_validation(db, limit=None)

    classes = onto.get_level(1)

    trn_X, trn_Y = get_xy(data_generator(trn_seq2pssm, trn_seq2go, classes))
    tst_X, tst_Y = get_xy(data_generator(tst_seq2pssm, tst_seq2go, classes))

    model = ModelCNN(classes)

    tst_shapes = list(zip(tst_X.keys(), tst_Y.keys()))
    for trnXshape, trnYshape in zip(trn_X.keys(), trn_Y.keys()):
        print(trn_X[trnXshape].shape, trn_Y[trnYshape].shape)
        tstXshape, tstYshape = tst_shapes[np.random.choice(len(tst_shapes))]
        model.fit(x=trn_X[trnXshape], y=trn_Y[trnYshape], batch_size=8, epochs=1, verbose=1,
                  callbacks=None, validation_data=[tst_X[tstXshape], tst_Y[tstYshape]])
