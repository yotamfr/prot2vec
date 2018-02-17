import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from src.python.consts import *

from src.python.preprocess import *

from src.python.geneontology import *

from src.python.baselines import *

from pymongo import MongoClient

from tqdm import tqdm

import tensorflow as tf
sess = tf.Session()

### Keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Embedding
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Concatenate, Flatten, Dropout
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LambdaCallback, LearningRateScheduler

from keras.losses import hinge

from keras import backend as K

K.set_session(sess)

import math

import argparse


LR = 0.032


def step_decay(epoch):
   initial_lrate = LR
   drop = 0.5
   epochs_drop = 1.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate


def Motifs(inpt):
    motif01 = Conv2D(2048, (1, 40), data_format='channels_first', padding='valid', activation='relu')(inpt)
    # motif03 = Conv2D(192, (3, 1), data_format='channels_first', padding='same', activation='relu')(motif01)
    # motif09 = Conv2D(64, (9, 1), data_format='channels_first', padding='same', activation='relu')(motif01)
    # motif18 = Conv2D(32, (18, 1), data_format='channels_first', padding='same', activation='relu')(motif01)
    # motif36 = Conv2D(16, (36, 1), data_format='channels_first', padding='same', activation='relu')(motif01)

    # return Concatenate(axis=1)([motif03, motif09, motif18, motif36])
    return motif01

def Features(motifs):
    feats = motifs
    feats = Conv2D(512, (9, 1), data_format='channels_first', activation='relu', padding='valid')(feats)
    # feats = MaxPooling2D((2, 1))(feats)
    # feats = Conv2D(128, (5, 1), data_format='channels_first', activation='relu', padding='valid')(feats)

    return GlobalMaxPooling2D(data_format='channels_first')(feats)


def Classifier(inpt, hidden_size, classes):
    x = inpt
    # We stack a deep densely-connected network on top
    x = Dense(hidden_size, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(hidden_size, activation='relu')(x)
    # x = Dropout(0.1)(x)

    # And finally we add the main logistic regression layer
    return Dense(len(classes), activation='tanh')(x)


def ModelCNN(classes):
    inp = Input(shape=(1, None, 40))
    out = Classifier(Features(Motifs(inp)), 512, classes)
    model = Model(inputs=[inp], outputs=[out])
    # sgd = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = optimizers.SGD(lr=LR, momentum=0.9, nesterov=True)
    model.compile(loss='hinge', optimizer=sgd)

    return model


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


def get_training_and_validation_streams(db, classes):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$lte": t0},
               'Aspect': ASPECT}
    seq2go_trn, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_train), db.goa_uniprot.count(q_train),
                                                 ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_trn.keys())).tolist()}, "pssm": {"$exists": True}}
    stream_trn = DataStream(db.pssm.find(query), db.pssm.count(query), seq2go_trn, classes)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": t0, "$lte": t1},
               'Aspect': ASPECT}

    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid), db.goa_uniprot.count(q_valid),
                                                 ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_tst.keys())).tolist()}, "pssm": {"$exists": True}}
    stream_tst = DataStream(db.pssm.find(query), db.pssm.count(query), seq2go_tst, classes)

    return stream_trn, stream_tst


def zeroone2oneminusone(vec):
    return np.add(np.multiply(np.array(vec), 2), -1)


def oneminusone2zeroone(vec):
    return np.divide(np.add(np.array(vec), 1), 2)


class DataStream(object):
    def __init__(self, source, count, seq2go, classes):

        self._classes = classes
        self._count = count
        self._source = source
        self._seq2go = seq2go

    def __iter__(self):

        classes = self._classes
        count = self._count
        source = self._source
        seq2go = self._seq2go

        s_cls = set(classes)

        for k, (seq, pssm, msa) in PssmCollectionLoader(source, count):
            if len(seq) < 30: continue
            y = np.zeros(len(classes))
            for go in onto.propagate(seq2go[k]):
                if go not in s_cls:
                    continue
                y[classes.index(go)] = 1

            if not pssm: continue

            x1 = [[AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)]] for i, aa in enumerate(seq)]
            x2 = msa

            yield k, x1, x2, y

    def __len__(self):
        return self._count


def data_generator(seq2pssm, seq2go, classes, verbose=1):
    s_cls = set(classes)
    for i, (k, v) in enumerate(seq2go.items()):

        if verbose == 1:
            sys.stdout.write("\r{0:.0f}%".format(100.0 * i / len(seq2go)))

        if k not in seq2pssm:
            continue
        y = np.zeros(len(classes))
        for go in onto.propagate(seq2go[k]):
            if go not in s_cls:
                continue
            y[classes.index(go)] = 1

        seq, pssm, msa = seq2pssm[k]
        if not pssm: continue

        x1 = [[AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)]] for i, aa in enumerate(seq)]
        x2 = msa

        yield k, x1, x2, y


def batch_generator(gen, normalize=False, batch_size=8):
    xByShapes, yByShapes = dict(), dict()
    for _, x, _, y in gen:
        x, y = np.array(x).reshape(1, len(x), 40), zeroone2oneminusone(y)
        if normalize: x = np.divide(np.add(x, -np.mean(x)), np.std(x))
        if x.shape in xByShapes:
            xByShapes[x.shape].append(x)
            yByShapes[x.shape].append(y)
            X, Y = xByShapes[x.shape], yByShapes[x.shape]
            if len(X) == batch_size:
                yield np.array(X), np.array(Y)
                del xByShapes[x.shape]
                del yByShapes[x.shape]

        else:
            xByShapes[x.shape] = [x]  # initially a list, because we're going to append items
            yByShapes[x.shape] = [y]

    for X, Y in zip(xByShapes.values(), yByShapes.values()):
        yield np.array(X), np.array(Y)


def eval_generator(model, gen_xy, length_xy, classes):

    pbar = tqdm(total=length_xy)
    i, m, n = 0, length_xy, len(classes)
    y_pred, y_true = np.zeros((m, n)), np.zeros((m, n))
    for i, (X, Y) in enumerate(gen_xy):
        assert len(X) == len(Y)
        k = len(Y)
        y_hat, y = model.predict(X), Y
        y_pred[i:i+k, ], y_true[i:i+k, ] = y_hat, y

        if (i + 1) % 20 == 0:
            loss = np.mean(hinge(y, y_hat).eval(session=sess))
            pbar.set_description("Validation Loss:%.5f" % loss)
        pbar.update(k)

    pbar.close()

    y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
    y_true = y_true[~np.all(y_true == 0, axis=1)]

    loss = np.mean(hinge(y_true, y_pred).eval(session=sess))
    y_pred = oneminusone2zeroone(y_pred)
    y_true = oneminusone2zeroone(y_true)
    f_max = F_max(y_pred, y_true, classes, np.arange(0.1, 1, 0.1))

    return y_true, y_pred, loss, f_max


def train_generator(model, gen_xy, length_xy, epoch, num_epochs, history=LossHistory(), lrate=LearningRateScheduler(step_decay)):

    pbar = tqdm(total=length_xy)

    for X, Y in gen_xy:
        assert len(X) == len(Y)

        model.fit(x=X, y=Y,
                  batch_size=8, epochs=num_epochs,
                  verbose=0,
                  validation_data=None,
                  initial_epoch=epoch,
                  callbacks=[history, lrate])
        pbar.set_description("Training Loss:%.5f" % np.mean(history.losses))
        pbar.update(len(Y))

    pbar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    ASPECT = 'F'  # Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(ASPECT)

    classes = onto.classes

    model = ModelCNN(classes)
    print(model.summary())

    sess = tf.Session()
    for epoch in range(args.num_epochs):
        trn_stream, tst_stream = get_training_and_validation_streams(db, classes)

        train_generator(model, batch_generator(trn_stream), len(trn_stream), epoch, args.num_epochs)
        _, _, loss, f_max = eval_generator(model, batch_generator(tst_stream), len(tst_stream), classes)

        print("[Epoch %d] (Validation Loss: %.5f, F_max: %.3f)" % (epoch + 1, loss, f_max))

        model_path = 'checkpoints/all-levels-cnn-%d-%.3f-%.2f.hdf5' % (epoch + 1, loss, f_max)
        model.save_weights(model_path)
