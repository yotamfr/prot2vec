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

BATCH_SIZE = 16

LONG_EXPOSURE = True


def step_decay(epoch):
    initial_lrate = LR
    drop = 0.5
    epochs_drop = 10.0
    lrate = max(0.0001, initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop)))
    return lrate


def batch_generator(data, onto, classes, batch_size=BATCH_SIZE, shuffle=True):

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
<<<<<<< HEAD
        seq = left + [AA.aa2index[aa] for aa in seq] + right
        return np.asarray(seq)

    def prepare_batch(sequences, labels):
        # b = max(max(map(len, sequences)), 100)
        b = MAX_LENGTH
=======
        seq = left + seq + right
        return np.asarray(seq)

    def prepare_batch(sequences, labels):
        b = max(max(map(len, sequences)), 100)
>>>>>>> 76ab7df4b4f8f2c4eb5b644154c84548fe4b40a3
        Y = np.asarray([labels2vec(lbl) for lbl in labels])
        X = np.asarray([pad_seq(seq, b) for seq in sequences])
        return X, Y

    indices = list(range(0, len(data), batch_size))
    if shuffle: np.random.shuffle(indices)
    while indices:
        ix = indices.pop()
        batch = data[ix: min(ix + batch_size, len(data))]
        ids, seqs, lbls = zip(*batch)
        yield ids, prepare_batch(seqs, lbls)


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        while len(self.losses) > 500:
            self.losses.pop(0)
        self.losses.append(logs.get('loss'))


def train(model, gen_xy, length_xy, epoch, num_epochs,
          history=LossHistory(), lrate=LearningRateScheduler(step_decay)):

    pbar = tqdm(total=length_xy)

    for _, (X, Y) in gen_xy:

        model.fit(x=X, y=Y,
                  batch_size=BATCH_SIZE,
                  epochs=max(num_epochs, epoch + 1) if LONG_EXPOSURE else epoch + 1,
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
    for i, (keys, (X, Y)) in enumerate(gen_xy):
        k = len(Y)
        ids.extend(keys)
        y_hat, y = model.predict(X), Y
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


def Inception(inpt, tower1=6, tower2=10, tower3=20):

    tower_0 = Conv1D(64, 1, padding='same', activation='relu')(inpt)

    tower_1 = Conv1D(64, 1, padding='same', activation='relu')(inpt)
    tower_1 = Conv1D(64, tower1, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(64, 1, padding='same', activation='relu')(inpt)
    tower_2 = Conv1D(64, tower2, padding='same', activation='relu')(tower_2)

    return Concatenate(axis=2)([tower_0, tower_1, tower_2])


def ResInception(inpt0, inpt1, tower1=6, tower2=10):
    incept0 = Inception(inpt0, tower1, tower2)
    incept1 = Inception(inpt1, tower1, tower2)
    return Concatenate(axis=2)([incept0, incept1])


def Features(inpt):
    feats = inpt
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
<<<<<<< HEAD
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
=======
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
>>>>>>> 76ab7df4b4f8f2c4eb5b644154c84548fe4b40a3
    feats = Dropout(0.3)(feats)
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    return feats


def ProteinCeption(classes, opt):
    inp = Input(shape=(None,))
    emb = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inp)
    incpt = Inception(Inception(Inception(emb)))
    out = Classifier(GlobalMaxPooling1D()(incpt), classes)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def DeeperSeq(classes, opt):
    inp = Input(shape=(None,))
    emb = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inp)
    feats = GlobalMaxPooling1D()(Features(emb))
    out = Classifier(feats, classes)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def MotifNet(classes, opt):
    inp = Input(shape=(None,))
    emb1 = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inp)
    inception = GlobalMaxPooling1D()(Inception(Inception(Inception(emb1))))
    emb2 = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inp)
    deeperseq = GlobalMaxPooling1D()(Features(emb2))
    out = Classifier(Concatenate()([inception, deeperseq]), classes)
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
    parser.add_argument("-e", "--eval_every", type=int, default=10,
                        help="How often to evaluate on the validation set.")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="How many epochs to train the model?")
    parser.add_argument('--long_exposure', action='store_true', default=False,
                        help="Train in LONG_EXPOSURE mode?")
    parser.add_argument("--arch", type=str, choices=['deepseq', 'inception', 'motifnet'],
                        default="inception", help="Specify the model arch.")


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

    if args.arch == "motifnet":
        model = MotifNet(classes, opt)
    elif args.arch == "inception":
        model = ProteinCeption(classes, opt)
    elif args.arch == "deepseq":
        model = DeeperSeq(classes, opt)
    else:
        print("Unknown arch")
        exit(0)

    if args.long_exposure:
        num_epochs = args.num_epochs // 10
        LONG_EXPOSURE = True
        BATCH_SIZE = 32
    else:
        num_epochs = args.num_epochs
        LONG_EXPOSURE = False
        BATCH_SIZE = 16

    if args.resume:
        model.load_weights(args.resume)
        print("Loaded model from disk")
    model.summary()

    print("Indexing Data...")
<<<<<<< HEAD
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, asp)
=======
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, asp, profile=0)
>>>>>>> 76ab7df4b4f8f2c4eb5b644154c84548fe4b40a3
    print("Loading Data...")
    trn_data = load_data(trn_stream)
    tst_data = load_data(tst_stream)

    for epoch in range(args.init_epoch, num_epochs):

        train(model, batch_generator(trn_data, onto, classes), len(trn_data), epoch, num_epochs)
        
        if epoch < num_epochs-1 and epoch % args.eval_every != 0:
            continue
        
        _, y_true, y_pred = predict(model, batch_generator(tst_data, onto, classes), len(tst_data), classes)
        loss, prs, rcs, f1s = evaluate(y_true, y_pred, classes)
        i = np.argmax(f1s)
        f_max = f1s[i]

        print("[Epoch %d/%d] (Validation Loss: %.5f, F_max: %.3f, precision: %.3f, recall: %.3f)"
              % (epoch + 1, num_epochs, loss, f1s[i], prs[i], rcs[i]))

        if f_max < 0.4: continue

        model_str = '%s-%d-%.5f-%.2f' % (args.arch, epoch + 1, loss, f_max)
        model.save_weights("checkpoints/%s.hdf5" % model_str)
        with open("checkpoints/%s.json" % model_str, "w+") as f:
            f.write(model.to_json())
