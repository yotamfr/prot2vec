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
from keras.layers import Input, Dense, Embedding, Activation
from keras.layers import Conv2D, Conv1D
from keras.layers import Dropout, BatchNormalization
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D
from keras.layers import Concatenate, Flatten, Reshape
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LambdaCallback, LearningRateScheduler

from keras.losses import hinge, binary_crossentropy

from keras import backend as K

from sklearn.metrics import log_loss

K.set_session(sess)

import math

import argparse

LR = 0.001

BATCH_SIZE = 32

t0 = datetime(2016, 2, 1, 0, 0)
t1 = TODAY

MAX_LENGTH = 2000
MIN_LENGTH = 1


def get_training_and_validation_streams(db, classes):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$lte": t0},
               'Aspect': ASPECT}
    seq2go_trn, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_train), db.goa_uniprot.count(q_train), ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_trn.keys())).tolist()}}
    stream_trn = DataStream(db.uniprot.find(query).batch_size(10), db.uniprot.count(query), seq2go_trn, classes)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": t0, "$lte": t1},
               'Aspect': ASPECT}

    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid), db.goa_uniprot.count(q_valid), ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_tst.keys())).tolist()}}
    stream_tst = DataStream(db.uniprot.find(query).batch_size(10), db.uniprot.count(query), seq2go_tst, classes)

    return stream_trn, stream_tst


def pad_seq(seq):
    seq += [PAD for _ in range(MAX_LENGTH - len(seq))]
    return np.asarray(seq)


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

        for k, seq in UniprotCollectionLoader(source, count):
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue
            y = np.zeros(len(classes))
            for go in onto.propagate(seq2go[k], include_root=False):
                if go not in s_cls:
                    continue
                y[classes.index(go)] = 1

                x = pad_seq([AA.aa2index[aa] for aa in seq])

            yield k, x, y

    def __len__(self):
        return self._count


def step_decay(epoch):
    initial_lrate = LR
    drop = 0.5
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


def Features(inpt):

    feats = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inpt)

    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = GlobalAveragePooling1D()(feats)
    return feats


def Classifier(inpt, classes):
    out = inpt
    out = Dense(len(classes), activation='linear')(out)
    out = BatchNormalization()(out)
    out = Activation('sigmoid')(out)
    return out


def ModelCNN(classes):
    inp = Input(shape=(None,))
    out = Classifier(Features(inp), classes)
    model = Model(inputs=[inp], outputs=[out])
    # sgd = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='binary_crossentropy', optimizer=adam)

    return model


def batch_generator(gen, normalize=False):
    xByShapes, yByShapes = dict(), dict()
    for _, x, y in gen:
        # x, y = np.array(x).reshape(1, len(x), 40), zeroone2oneminusone(y)
        if normalize: x = np.divide(np.add(x, -np.mean(x)), np.std(x))
        if x.shape in xByShapes:
            xByShapes[x.shape].append(x)
            yByShapes[x.shape].append(y)
            X, Y = xByShapes[x.shape], yByShapes[x.shape]
            if len(X) == BATCH_SIZE:
                yield np.array(X), np.array(Y)
                del xByShapes[x.shape]
                del yByShapes[x.shape]
        else:
            xByShapes[x.shape] = [x]
            yByShapes[x.shape] = [y]

    for X, Y in zip(xByShapes.values(), yByShapes.values()):
        yield np.array(X), np.array(Y)


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="How many epochs to train the model?")


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train_model(model, gen_xy, length_xy, epoch, num_epochs,
                history=LossHistory(), lrate=LearningRateScheduler(step_decay)):

    pbar = tqdm(total=length_xy)

    for X, Y in gen_xy:
        assert len(X) == len(Y)

        model.fit(x=X, y=Y,
                  batch_size=BATCH_SIZE,
                  epochs=num_epochs,
                  verbose=0,
                  validation_data=None,
                  initial_epoch=epoch,
                  callbacks=[history])
        pbar.set_description("Training Loss:%.5f" % np.mean(history.losses))
        pbar.update(len(Y))

    pbar.close()


def zeroone2oneminusone(vec):
    return np.add(np.multiply(np.array(vec), 2), -1)


def oneminusone2zeroone(vec):
    return np.divide(np.add(np.array(vec), 1), 2)


def calc_loss(y_true, y_pred):
    return np.mean([log_loss(y, y_hat) for y, y_hat in zip(y_true, y_pred) if np.any(y)])


def eval_model(model, gen_xy, length_xy, classes):
    pbar = tqdm(total=length_xy, desc="Evaluating Loss")
    i, m, n = 0, length_xy, len(classes)
    y_pred, y_true = np.zeros((m, n)), np.zeros((m, n))
    for i, (X, Y) in enumerate(gen_xy):
        assert len(X) == len(Y)
        k = len(Y)
        y_hat, y = model.predict(X), Y
        y_pred[i:i + k, ], y_true[i:i + k, ] = y_hat, y
        pbar.update(k)

    pbar.close()

    y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
    y_true = y_true[~np.all(y_true == 0, axis=1)]

    f_max = F_max(y_pred, y_true, classes, np.arange(0.1, 1, 0.1))

    return y_true, y_pred, calc_loss(y_true, y_pred), f_max


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
    classes.remove(onto.root)
    assert onto.root not in classes

    model = ModelCNN(classes)
    model.summary()

    sess = tf.Session()
    for epoch in range(args.num_epochs):

        trn_stream, tst_stream = get_training_and_validation_streams(db, classes)

        train_model(model, batch_generator(trn_stream), len(trn_stream), epoch, args.num_epochs)
        _, _, loss, f_max = eval_model(model, batch_generator(tst_stream), len(tst_stream), classes)

        print("[Epoch %d] (Validation Loss: %.5f, F_max: %.3f)" % (epoch + 1, loss, f_max))

        model_path = 'checkpoints/deep-seq-%d-%.3f-%.2f.hdf5' % (epoch + 1, loss, f_max)
        model.save_weights(model_path)
