import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from src.python.consts import *

from src.python.preprocess import *

from src.python.geneontology import *

from src.python.baselines import *

from sklearn.metrics import hinge_loss

from pymongo import MongoClient

from tqdm import tqdm

import tensorflow as tf
sess = tf.Session()

### Keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Concatenate, Flatten, Reshape
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LambdaCallback, LearningRateScheduler

from keras.losses import hinge

from keras import backend as K

K.set_session(sess)

import math

import argparse


LR = 0.032


def _get_labeled_data(db, query, limit, pssm=True):

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    seqid2goid, _ = GoAnnotationCollectionLoader(s, c, ASPECT).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}, "pssm": {"$exists": True}}

    if pssm:
        num_seq = db.pssm.count(query)
        src_seq = db.pssm.find(query)
        seqid2seq = PssmCollectionLoader(src_seq, num_seq).load()
    else:
        num_seq = db.uniprot.count(query)
        src_seq = db.uniprot.find(query)
        seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    seqid2goid = {k: v for k, v in seqid2goid.items() if len(v) > 1 or 'GO:0005515' not in v}
    seqid2seq = {k: v for k, v in seqid2seq.items() if k in seqid2goid and len(v[0]) > 30}
    seqid2goid = {k: v for k, v in seqid2goid.items() if k in seqid2seq}

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
        x1 = [[AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)]] for i, aa in enumerate(seq)]
        x2 = msa

        yield k, x1, x2, y


def step_decay(epoch):
   initial_lrate = LR
   drop = 0.5
   epochs_drop = 5.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate


def Motifs(inpt):
    motif01 = Conv2D(192, (1, 40), padding='valid', activation='relu')(inpt)
    motif03 = Conv2D(64, (3, 1), padding='same', activation='relu')(motif01)
    motif09 = Conv2D(64, (9, 1), padding='same', activation='relu')(motif01)
    motif18 = Conv2D(64, (18, 1), padding='same', activation='relu')(motif01)
    motif36 = Conv2D(64, (36, 1), padding='same', activation='relu')(motif01)

    return Concatenate(axis=1)([motif03, motif09, motif18, motif36])


def Features(inpt):
    out = inpt
    out = Conv2D(64, (3, 1), activation='relu', padding='same')(out)
    out = Conv2D(64, (3, 1), activation='relu', padding='same')(out)
    out = MaxPooling2D((2, 1))(out)
    out = Conv2D(128, (3, 1), activation='relu', padding='same')(out)
    out = Conv2D(128, (3, 1), activation='relu', padding='same')(out)

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
    inp = Input(shape=(1, None, 40))
    out = Classifier(Features(Motifs(inp)), 64, classes)
    model = Model(inputs=[inp], outputs=[out])
    # sgd = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = optimizers.SGD(lr=LR, momentum=0.9, nesterov=True)
    model.compile(loss='hinge', optimizer=sgd)

    return model


def get_xy(gen, normalize=False):
    xByShapes, yByShapes = dict(), dict()
    for _, x, _, y in gen:
        x, y = np.array(x).reshape(1, len(x), 40), zeroone2oneminusone(y)
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


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="How many epochs to train the model?")


class LossHistory(Callback):

    def __init__(self):
        self.losses = []

    # def on_train_begin(self, logs={}):
    #     self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train(model, X, Y, epoch, num_epochs, history=LossHistory(), lrate=LearningRateScheduler(step_decay)):
    m = sum(map(lambda k: len(Y[k]), Y.keys()))
    pbar = tqdm(total=m)
    for x_shp, y_shp in zip(X.keys(), Y.keys()):
        model.fit(x=trn_X[x_shp], y=trn_Y[y_shp],
                  batch_size=8, epochs=num_epochs,
                  verbose=0,
                  validation_data=None,
                  initial_epoch=epoch,
                  callbacks=[history, lrate])
        pbar.set_description("Training Loss:%.5f" % np.mean(history.losses))
        pbar.update(len(Y[y_shp]))

    pbar.close()


def zeroone2oneminusone(vec):
    return np.add(np.multiply(np.array(vec), 2), -1)


def oneminusone2zeroone(vec):
    return np.divide(np.add(np.array(vec), 1), 2)


def evaluate(model, X, Y, classes):
    i, m, n = 0, sum(map(lambda k: len(Y[k]), Y.keys())), len(classes)
    y_pred, y_true = np.zeros((m, n)), np.zeros((m, n))
    for x_shp, y_shp in zip(X.keys(), Y.keys()):
        k = len(Y[y_shp])
        y_hat, y = model.predict(X[x_shp]), Y[y_shp]
        y_pred[i:i+k, ], y_true[i:i+k, ] = y_hat, y
        i += k
    loss = np.mean(hinge(y_true, y_pred).eval(session=sess))
    y_pred = oneminusone2zeroone(y_pred)
    y_true = oneminusone2zeroone(y_true)
    f_max = F_max(y_pred, y_true, classes)

    return y_true, y_pred, loss, f_max


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    ASPECT = 'F'  # Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(ASPECT)

    trn_seq2pssm, trn_seq2go, tst_seq2pssm, tst_seq2go = load_training_and_validation(db, limit=None)

    train_and_validation_data = ({k: v[0] for k, v in trn_seq2pssm.items()}, trn_seq2go,
                                 {k: v[0] for k, v in tst_seq2pssm.items()}, tst_seq2go)

    classes = onto.get_level(1)

    trn_X, trn_Y = get_xy(data_generator(trn_seq2pssm, trn_seq2go, classes))
    tst_X, tst_Y = get_xy(data_generator(tst_seq2pssm, tst_seq2go, classes))

    model = ModelCNN(classes)
    print(model.summary())

    model_path = 'checkpoints/1st-level-cnn-{epoch:03d}-{val_loss:.3f}.hdf5'

    # callbacks = [
    #
    #     EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    #
    #     ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
    #
    # ]

    # pred_blast, perf_blast = evaluate_performance(db, ["blast"], ASPECT, train_and_validation_data, load_file=1, plot=0)
    # print("\nBLAST Validation F_max: %.3f\n" % max(perf_blast["blast"][2]))

    sess = tf.Session()
    for epoch in range(args.num_epochs):
        train(model, trn_X, trn_Y, epoch, args.num_epochs)
        _, _, loss, f_max = evaluate(model, tst_X, tst_Y, classes)
        print("[Epoch %d] (Validation Loss: %.5f, F_max: %.3f)" % (epoch + 1, loss, f_max))
        # tst_shapes = list(zip(tst_X.keys(), tst_Y.keys()))
        # for trnXshape, trnYshape in zip(trn_X.keys(), trn_Y.keys()):
        #     print(trn_X[trnXshape].shape, trn_Y[trnYshape].shape)
        #     tstXshape, tstYshape = tst_shapes[np.random.choice(len(tst_shapes))]
        #     model.fit(x=trn_X[trnXshape],
        #               y=trn_Y[trnYshape],
        #               batch_size=8, epochs=1,
        #               verbose=0,
        #               validation_data=[tst_X[tstXshape], tst_Y[tstYshape]])
