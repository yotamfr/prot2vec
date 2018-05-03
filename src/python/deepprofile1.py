import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from src.python.baselines import *

from src.python.preprocess2 import *

from pymongo import MongoClient

from tqdm import tqdm

import tensorflow as tf

### Keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Activation
from keras.layers import Conv2D, Conv1D
from keras.layers import Dropout, BatchNormalization
from keras.layers import MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Concatenate, Flatten, Reshape
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LambdaCallback, LearningRateScheduler
# from keras.losses import hinge, binary_crossentropy

from keras import backend as K

from sklearn.metrics import log_loss

import math

import argparse

sess = tf.Session()
K.set_session(sess)

LR = 0.001

BATCH_SIZE = 32

LONG_EXPOSURE = True


def step_decay(epoch):
    initial_lrate = LR
    drop = 0.5
    epochs_drop = 20.0
    lrate = max(0.0001, initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop)))
    return lrate


def batch_generator(data, onto, classes):

    s_cls = set(classes)

    def labels2vec(lbl):
        y = np.zeros(len(classes))
        for go in onto.propagate(lbl, include_root=False):
            if go not in s_cls:
                continue
            y[classes.index(go)] = 1
        return y

    def pad_seq(seq, max_length=MAX_LENGTH):
        delta = max_length - len(seq)
        left = [PAD for _ in range(delta // 2)]
        right = [PAD for _ in range(delta - delta // 2)]
        seq = left + seq + right
        return np.asarray(seq)

    def pad_pssm(pssm, max_length=MAX_LENGTH):
        delta = max_length - len(pssm)
        left = [[0] * 40 for _ in range(delta // 2)]
        right = [[0] * 40 for _ in range(delta - delta // 2)]
        pssm = left + pssm + right
        return np.asarray(pssm)

    def prepare_batch(sequences, pssms, labels):
        b = max(map(len, sequences)) + 100
        Y = np.asarray([labels2vec(lbl) for lbl in labels])
        X1 = np.asarray([pad_seq(seq, b) for seq in sequences])
        X2 = np.asarray([pad_pssm(pssm, b) for pssm in pssms])
        return X1, X2, Y

    batch = []
    for packet in data:
        if len(batch) == BATCH_SIZE:
            ids, seqs, pssms, lbls = zip(*batch)
            yield ids, prepare_batch(seqs, pssms, lbls)
            batch = []
        batch.append(packet)
    ids, seqs, pssms, lbls = zip(*batch)
    yield ids, prepare_batch(seqs, pssms, lbls)     # Last batch


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train(model, gen_xy, length_xy, epoch, num_epochs,
          history=LossHistory(), lrate=LearningRateScheduler(step_decay)):

    pbar = tqdm(total=length_xy)

    for _, (X1, X2, Y) in gen_xy:

        model.fit(x=X2, y=Y,
                  batch_size=BATCH_SIZE,
                  epochs=num_epochs if LONG_EXPOSURE else epoch + 1,
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


def calc_loss(y_true, y_pred):
    return np.mean([log_loss(y, y_hat) for y, y_hat in zip(y_true, y_pred) if np.any(y)])


def predict(model, gen_xy, length_xy, classes):
    pbar = tqdm(total=length_xy, desc="Predicting...")
    i, m, n = 0, length_xy, len(classes)
    ids = list()
    y_pred, y_true = np.zeros((m, n)), np.zeros((m, n))
    for i, (keys, (X1, X2, Y)) in enumerate(gen_xy):
        k = len(Y)
        ids.extend(keys)
        y_hat, y = model.predict(X2), Y
        y_pred[i:i + k, ] = y_hat
        y_true[i:i + k, ] = y
        pbar.update(k)
    pbar.close()
    return ids, y_true, y_pred


def evaluate(y_true, y_pred, classes):
    y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
    y_true = y_true[~np.all(y_true == 0, axis=1)]
    prs, rcs, f1s = performance(y_pred, y_true, classes)
    return calc_loss(y_true, y_pred), prs, rcs, f1s


def Classifier(inp1d, classes):
    out = Dense(len(classes))(inp1d)
    out = BatchNormalization()(out)
    out = Activation('sigmoid')(out)
    return out


def Inception(inpt, tower1=6, tower2=10):

    tower_1 = Conv1D(64, 1, padding='same', activation='relu')(inpt)
    tower_1 = Conv1D(64, tower1, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(64, 1, padding='same', activation='relu')(inpt)
    tower_2 = Conv1D(64, tower2, padding='same', activation='relu')(tower_2)

    # tower_3 = MaxPooling1D(3, strides=1, padding='same')(inpt)
    # tower_3 = Conv1D(64, 1, padding='same', activation='relu')(tower_3)

    return Concatenate(axis=2)([tower_1, tower_2])


def Features(inpt):
    feats = inpt
    feats = BatchNormalization()(feats)
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = GlobalMaxPooling1D()(feats)
    return feats


def DeepProfile(classes, opt):
    inp = Input(shape=(None, 40))
    out = Classifier(Features(inp), classes)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB"),
    parser.add_argument("--aspect", type=str, choices=['F', 'P', 'C'],
                        default="F", help="Specify the ontology aspect.")
    parser.add_argument("--init_epoch", type=int, default=0,
                        help="Which epoch to start training the model?")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="How many epochs to train the model?")
    parser.add_argument('--long_exposure', action='store_true', default=False,
                        help="Train in LONG_EXPOSURE mode?")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    asp = args.aspect  # default: Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(asp)

    classes = onto.classes
    classes.remove(onto.root)
    assert onto.root not in classes

    opt = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model = DeepProfile(classes, opt)
    LONG_EXPOSURE = args.long_exposure

    if args.resume:
        model.load_weights(args.resume)
        print("Loaded model from disk")
    model.summary()

    print("Indexing Data...")
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, asp)
    print("Loading Data...")
    trn_data = load_data(trn_stream)
    tst_data = load_data(tst_stream)

    for epoch in range(args.init_epoch, args.num_epochs):

        train(model, batch_generator(trn_data, onto, classes), len(trn_data), epoch, args.num_epochs)
        _, y_true, y_pred = predict(model, batch_generator(tst_data, onto, classes), len(tst_data), classes)
        loss, prs, rcs, f1s = evaluate(y_true, y_pred, classes)
        i = np.argmax(f1s)
        f_max = f1s[i]

        print("[Epoch %d/%d] (Validation Loss: %.5f, F_max: %.3f, precision: %.3f, recall: %.3f)"
              % (epoch + 1, args.num_epochs, loss, f1s[i], prs[i], rcs[i]))

        if f_max < 0.4: continue

        model_str = '%s-%d-%.5f-%.2f' % ("deepprofile", epoch + 1, loss, f_max)
        model.save_weights("checkpoints/%s.hdf5" % model_str)
        with open("checkpoints/%s.json" % model_str, "w+") as f:
            f.write(model.to_json())
