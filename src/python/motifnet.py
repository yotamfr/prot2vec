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

t0 = datetime(2014, 1, 1, 0, 0)
t1 = datetime(2014, 9, 1, 0, 0)

MAX_LENGTH = 2000
MIN_LENGTH = 1


def get_training_and_validation_streams(db, limit=None):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$lte": t0},
               'Aspect': ASPECT}
    seq2go_trn, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_train), db.goa_uniprot.count(q_train), ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_trn.keys())).tolist()}}
    count = limit if limit else db.uniprot.count(query)
    source = db.uniprot.find(query).batch_size(10)
    if limit: source = source.limit(limit)
    stream_trn = DataStream(source, count, seq2go_trn)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": t0, "$lte": t1},
               'Aspect': ASPECT}

    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid), db.goa_uniprot.count(q_valid), ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_tst.keys())).tolist()}}
    count = limit if limit else db.uniprot.count(query)
    source = db.uniprot.find(query).batch_size(10)
    if limit: source = source.limit(limit)
    stream_tst = DataStream(source, count, seq2go_tst)

    return stream_trn, stream_tst


class DataStream(object):
    def __init__(self, source, count, seq2go):

        self._count = count
        self._source = source
        self._seq2go = seq2go

    def __iter__(self):

        count = self._count
        source = self._source
        seq2go = self._seq2go

        for k, seq in UniprotCollectionLoader(source, count):
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue

            x = [AA.aa2index[aa] for aa in seq]

            yield k, x, seq2go[k]

    def __len__(self):
        return self._count


def step_decay(epoch):
    initial_lrate = LR
    drop = 0.5
    epochs_drop = 1.0
    lrate = max(0.0001, initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop)))
    return lrate


def OriginalIception(inpt, num_channels=64):

    # tower_0 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)

    tower_1 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_1 = Conv1D(num_channels, 3, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_2 = Conv1D(num_channels, 5, padding='same', activation='relu')(tower_2)

    # tower_3 = MaxPooling1D(3, padding='same')(inpt)
    # tower_3 = Conv1D(num_channels, 1, padding='same')(tower_3)

    return Concatenate(axis=2)([tower_1, tower_2,])


def LargeInception(inpt, num_channels=64):

    tower_1 = Conv1D(num_channels, 6, padding='same', activation='relu')(inpt)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Conv1D(num_channels, 6, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(num_channels, 10, padding='same', activation='relu')(inpt)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Conv1D(num_channels, 10, padding='same', activation='relu')(tower_2)

    return Concatenate(axis=2)([tower_1, tower_2])


def SmallInception(inpt, num_channels=150):

    tower_1 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_1 = Conv1D(num_channels, 5, padding='same', activation='relu')(tower_1)
    # tower_1 = BatchNormalization()(tower_1)

    tower_2 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_2 = Conv1D(num_channels, 15, padding='same', activation='relu')(tower_2)
    # tower_2 = BatchNormalization()(tower_2)

    return Concatenate(axis=2)([tower_1, tower_2])


def Classifier(inp1d, classes):
    out = Dense(len(classes))(inp1d)
    out = BatchNormalization()(out)
    out = Activation('sigmoid')(out)
    return out


def MotifNet(classes, opt):
    inpt = Input(shape=(None,))
    out = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inpt)
    out = Conv1D(250, 15, activation='relu', padding='valid')(out)
    out = Dropout(0.2)(out)
    out = Conv1D(100, 15, activation='relu', padding='valid')(out)
    out = SmallInception(out)
    out = Dropout(0.2)(out)
    out = SmallInception(out)
    out = Dropout(0.2)(out)
    out = Conv1D(250, 5, activation='relu', padding='valid')(out)
    out = Dropout(0.2)(out)
    out = Classifier(GlobalMaxPooling1D()(out), classes)
    model = Model(inputs=[inpt], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def Inception(inpt, tower1=6, tower2=10):

    tower_1 = Conv1D(64, 1, padding='same', activation='relu')(inpt)
    tower_1 = Conv1D(64, tower1, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(64, 1, padding='same', activation='relu')(inpt)
    tower_2 = Conv1D(64, tower2, padding='same', activation='relu')(tower_2)

    # tower_3 = MaxPooling1D(3, strides=1, padding='same')(inpt)
    # tower_3 = Conv1D(64, 1, padding='same', activation='relu')(tower_3)

    return Concatenate(axis=2)([tower_1, tower_2])


def ProteinInception(classes, opt):
    inpt = Input(shape=(None,))
    img = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inpt)
    feats = Inception(Inception(img))
    out = Classifier(GlobalMaxPooling1D()(feats), classes)
    model = Model(inputs=[inpt], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


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
    feats = GlobalMaxPooling1D()(feats)
    return feats


def Classifier(inpt, classes):
    out = inpt
    out = Dense(len(classes), activation='linear')(out)
    out = BatchNormalization()(out)
    out = Activation('sigmoid')(out)
    return out


def DeeperSeq(classes, opt):
    inp = Input(shape=(None,))
    out = Classifier(Features(inp), classes)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def batch_generator(stream, onto, classes):

    s_cls = set(classes)
    data = dict()

    def labels2vec(lbl):
        y = np.zeros(len(classes))
        for go in onto.propagate(lbl, include_root=False):
            if go not in s_cls:
                continue
            y[classes.index(go)] = 1

    def pad_seq(seq, max_length=MAX_LENGTH):
        delta = max_length - len(seq)
        seq = [PAD for _ in range(delta - delta // 2)] + seq + [PAD for _ in range(delta // 2)]
        return np.asarray(seq)

    def prepare_batch(sequences, labels):
        # b = max(map(len, sequences))
        Y = np.asarray([e for e in map(labels2vec, labels)])
        X = np.asarray([e for e in map(pad_seq, sequences)])
        return X, Y

    for k, x, y in stream:
        if len(x) in data:
            data[len(x)].append([k, x, y])
            ids, seqs, lbls = zip(*data[len(x)])
            if len(seqs) == BATCH_SIZE:
                yield ids, prepare_batch(seqs, lbls)
                del data[len(x)]
        else:
            data[len(x)] = [[k, x, y]]

    for ids, seqs, lbls in zip(*data.values()):
        yield ids, prepare_batch(seqs, lbls)


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train(model, gen_xy, length_xy, epoch, num_epochs,
          history=LossHistory(), lrate=LearningRateScheduler(step_decay)):

    pbar = tqdm(total=length_xy)

    for _, (X, Y) in gen_xy:

        model.fit(x=X, y=Y,
                  batch_size=BATCH_SIZE,
                  epochs=num_epochs if LONG_EXPOSURE else epoch + 1,
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


def predict(model, gen_xy, length_xy, classes):
    pbar = tqdm(total=length_xy, desc="Predicting...")
    i, m, n = 0, length_xy, len(classes)
    ids = list()
    y_pred, y_true = np.zeros((m, n)), np.zeros((m, n))
    for i, (keys, (X, Y)) in enumerate(gen_xy):
        k = len(Y)
        ids.extend(keys)
        y_hat, y = model.predict(X), Y
        y_pred[i:i + k, ], y_true[i:i + k, ] = y_hat, y
        pbar.update(k)
    pbar.close()
    return ids, y_true, y_pred


def evaluate(y_true, y_pred, classes):
    y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
    y_true = y_true[~np.all(y_true == 0, axis=1)]
    prs, rcs, f1s = performance(y_pred, y_true, classes)
    return calc_loss(y_true, y_pred), prs, rcs, f1s


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB"),
    parser.add_argument("--aspect", type=str, choices=['F', 'P', 'C'],
                        default="F", help="Specify the ontology aspect.")
    parser.add_argument("--init_epoch", type=int, default=0,
                        help="Which epoch to start training the model?")
    parser.add_argument("--arch", type=str, choices=['deeperseq', 'motifnet', 'inception'],
                        default="inception", help="Specify the model arch.")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    ASPECT = args.aspect  # default: Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(ASPECT)

    classes = onto.classes
    classes.remove(onto.root)
    assert onto.root not in classes

    opt = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    if args.arch == 'inception':
        model = ProteinInception(classes, opt)
        LONG_EXPOSURE = False
        num_epochs = 200
    elif args.arch == 'deeperseq':
        model = DeeperSeq(classes, opt)
        LONG_EXPOSURE = True
        num_epochs = 20
    elif args.arch == 'motifnet':
        model = MotifNet(classes, opt)
        LONG_EXPOSURE = False
        num_epochs = 200
    else:
        print('Unknown model arch')
        exit(0)

    if args.resume:
        model.load_weights(args.resume)
        print("Loaded model from disk")
    model.summary()

    for epoch in range(args.init_epoch, num_epochs):

        trn_stream, tst_stream = get_training_and_validation_streams(db)

        train(model, batch_generator(trn_stream, onto, classes), len(trn_stream), epoch, num_epochs)
        y_true, y_pred = predict(model, batch_generator(tst_stream, onto, classes), len(tst_stream), classes)
        loss, prs, rcs, f1s = evaluate(y_true, y_pred, classes)
        i = np.argmax(f1s)
        f_max = f1s[i]

        print("[Epoch %d/%d] (Validation Loss: %.5f, F_max: %.3f, precision: %.3f, recall: %.3f)"
              % (epoch + 1, num_epochs, loss, f1s[i], prs[i], rcs[i]))

        model_path = 'checkpoints/%s-%d-%.5f-%.2f' % (args.arch, epoch + 1, loss, f_max)
        model.save_weights("%s.hdf5" % model_path)
        with open("%s.json" % model_path, "w+") as f:
            f.write(model.to_json())
