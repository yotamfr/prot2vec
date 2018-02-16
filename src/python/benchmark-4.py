import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


def get_training_and_validation_streams(db, class_levels):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$lte": t0},
               'Aspect': ASPECT}
    seq2go_trn, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_train), db.goa_uniprot.count(q_train),
                                                 ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_trn.keys())).tolist()}, "pssm": {"$exists": True}}
    stream_trn = DataStream(db.pssm.find(query), db.pssm.count(query), seq2go_trn, class_levels)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": t0, "$lte": t1},
               'Aspect': ASPECT}

    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid), db.goa_uniprot.count(q_valid),
                                                 ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_tst.keys())).tolist()}, "pssm": {"$exists": True}}
    stream_tst = DataStream(db.pssm.find(query), db.pssm.count(query), seq2go_tst, class_levels)

    return stream_trn, stream_tst


class DataStream(object):
    def __init__(self, source, count, seq2go, class_levels):

        self._count = count
        self._source = source
        self._seq2go = seq2go
        self._clss = class_levels

    def __iter__(self):

        clss = self._clss
        count = self._count
        source = self._source
        seq2go = self._seq2go

        for k, (seq, pssm, msa) in PssmCollectionLoader(source, count):
            if len(seq) < 30: continue

            ys = []
            for classes in clss:
                s_cls = set(classes)
                y = np.zeros(len(classes))
                for go in onto.propagate(seq2go[k]):
                    if go not in s_cls:
                        continue
                    y[classes.index(go)] = 1
                ys.append(y)

            if not pssm: continue

            x1 = [[AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)]] for i, aa in enumerate(seq)]
            x2 = msa

            yield k, x1, x2, ys

    def __len__(self):
        return self._count


def step_decay(epoch):
    initial_lrate = LR
    drop = 0.5
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


def Motifs(inpt):
    motif01 = Conv2D(1024, (1, 40), data_format='channels_first', padding='valid', activation='relu')(inpt)
    # motif03 = Conv2D(192, (3, 1), data_format='channels_first', padding='same', activation='relu')(motif01)
    # motif09 = Conv2D(64, (9, 1), data_format='channels_first', padding='same', activation='relu')(motif01)
    # motif18 = Conv2D(32, (18, 1), data_format='channels_first', padding='same', activation='relu')(motif01)
    # motif36 = Conv2D(16, (36, 1), data_format='channels_first', padding='same', activation='relu')(motif01)

    # return Concatenate(axis=1)([motif03, motif09, motif18, motif36])
    return motif01


def Features(motifs):
    feats = motifs
    feats = Conv2D(192, (9, 1), data_format='channels_first', activation='relu', padding='valid')(feats)
    # feats = MaxPooling2D((2, 1))(feats)
    # feats = Conv2D(128, (5, 1), data_format='channels_first', activation='relu', padding='valid')(feats)

    return GlobalMaxPooling2D(data_format='channels_first')(feats)


def Classifier(inpt1, inpt2, hidden_size, classes):
    if inpt2 is not None:
        x1, x2 = inpt1, inpt2
        x = Concatenate(axis=1)([x1, x2])
    else:
        x = inpt1
    # We stack a deep densely-connected network on top
    x = Dense(hidden_size * 2, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(hidden_size, activation='relu')(x)
    # x = Dropout(0.1)(x)

    # And finally we add the main logistic regression layer
    return Dense(len(classes), activation='tanh')(x)


def ModelCNN(clss):
    inp = Input(shape=(1, None, 40))
    cls1, cls2 = clss
    out1 = Classifier(Features(Motifs(inp)), None, 64, cls1)
    out2 = Classifier(Features(Motifs(inp)), out1, 64, cls2)
    model = Model(inputs=[inp], outputs=[out1, out2])
    # sgd = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = optimizers.SGD(lr=LR, momentum=0.9, nesterov=True)
    model.compile(loss='hinge', optimizer=sgd)

    return model


def batch_generator(gen, normalize=False, batch_size=8):
    xByShapes, yByShapes = dict(), dict()
    for _, x, _, ys in gen:
        x, ys = np.array(x).reshape(1, len(x), 40), zeroone2oneminusone(ys)
        if normalize: x = np.divide(np.add(x, -np.mean(x)), np.std(x))
        if x.shape in xByShapes:
            xByShapes[x.shape].append(x)
            yByShapes[x.shape].append(ys)
            X, Y = xByShapes[x.shape], yByShapes[x.shape]
            if len(X) == batch_size:
                yield np.array(X), np.array(Y)
                del xByShapes[x.shape]
                del yByShapes[x.shape]

        else:
            xByShapes[x.shape] = [x]  # initially a list, because we're going to append items
            yByShapes[x.shape] = [ys]

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

    # def on_train_begin(self, logs={}):
    #     self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train_generator(model, gen_xy, length_xy, epoch, num_epochs, history=LossHistory(),
                    lrate=LearningRateScheduler(step_decay)):
    pbar = tqdm(total=length_xy)

    for X, Y in gen_xy:
        assert len(X) == len(Y)

        model.fit(x=[X], y=[np.asarray(Y) for Y in zip(*Y)],
                  batch_size=8, epochs=num_epochs,
                  verbose=0,
                  validation_data=None,
                  initial_epoch=epoch,
                  callbacks=[history, lrate])
        pbar.set_description("Training Loss:%.5f" % np.mean(history.losses))
        pbar.update(len(Y))

    pbar.close()


def zeroone2oneminusone(vec):
    return np.add(np.multiply(np.array(vec), 2), -1)


def oneminusone2zeroone(vec):
    return np.divide(np.add(np.array(vec), 1), 2)


def eval_generator(model, gen_xy, length_xy, clss):

    classes = reduce(lambda x, y: x + y, clss, [])

    pbar = tqdm(total=length_xy)
    y_true = np.zeros((length_xy, len(classes)))
    y_pred = np.zeros((length_xy, len(classes)))

    for i, (X, Ys) in enumerate(gen_xy):
        k = len(X)
        for j, Y in enumerate(zip(*Ys)):
            assert len(X) == len(Y)
            y_hat, y = model.predict(X), Y
            s = sum([len(cls) for cls in clss[:j]])
            e = s+len(clss[j])
            y_pred[i:i+k, s:e], y_true[i:i+k, s:e] = y_hat, y

        if (i + 1) % 100 == 0:
            loss = np.mean(hinge(y_true, y_pred).eval(session=sess))
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    ASPECT = 'F'  # Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(ASPECT)

    clss = [onto.get_level(1), onto.get_level(2)]

    model = ModelCNN(clss)
    print(model.summary())

    model_path = 'checkpoints/1st-level-cnn-{epoch:03d}-{val_loss:.3f}.hdf5'

    sess = tf.Session()
    for epoch in range(args.num_epochs):

        trn_stream, tst_stream = get_training_and_validation_streams(db, clss)

        train_generator(model, batch_generator(trn_stream), len(trn_stream), epoch, args.num_epochs)
        _, _, loss, f_max = eval_generator(model, batch_generator(tst_stream), len(tst_stream), clss)

        print("[Epoch %d] (Validation Loss: %.5f, F_max: %.3f)" % (epoch + 1, loss, f_max))

        model_path = 'checkpoints/1st-level-cnn-%d-%.3f-%.2f.hdf5' % (epoch + 1, loss, f_max)
        model.save_weights(model_path)
