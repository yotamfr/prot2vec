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
from keras.layers import MaxPooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D
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

t0 = datetime(2014, 1, 1, 0, 0)
t1 = datetime(2014, 9, 1, 0, 0)

MAX_LENGTH = 2000
MIN_LENGTH = 1


def get_classes(db, onto, start=t0, end=t1):

    q1 = {'DB': 'UniProtKB',
         'Evidence': {'$in': exp_codes},
         'Date': {"$lte": start},
         'Aspect': ASPECT}
    q2 = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": start, "$lte": end},
               'Aspect': ASPECT}

    def helper(q):
        seq2go, _ = GoAnnotationCollectionLoader(
            db.goa_uniprot.find(q), db.goa_uniprot.count(q), ASPECT).load()
        for i, (k, v) in enumerate(seq2go.items()):
            sys.stdout.write("\r{0:.0f}%".format(100.0 * i / len(seq2go)))
            seq2go[k] = onto.propagate(v)
        return reduce(lambda x, y: set(x) | set(y), seq2go.values(), set())

    return list(helper(q1) | helper(q2))


def get_training_and_validation_streams(db, onto, classes, limit=None, start=t0, end=t1):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$lte": start},
               'Aspect': ASPECT}
    seq2go_trn, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_train), db.goa_uniprot.count(q_train), ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_trn.keys())).tolist()}}
    count = limit if limit else db.uniprot.count(query)
    source = db.uniprot.find(query).batch_size(10)
    if limit: source = source.limit(limit)
    stream_trn = DataStream(source, count, seq2go_trn)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": start, "$lte": end},
               'Aspect': ASPECT}

    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid), db.goa_uniprot.count(q_valid), ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_tst.keys())).tolist()}}
    count = limit if limit else db.uniprot.count(query)
    source = db.uniprot.find(query).batch_size(10)
    if limit: source = source.limit(limit)
    stream_tst = DataStream(source, count, seq2go_tst)

    return stream_trn, stream_tst


def pad_seq(seq, max_length=MAX_LENGTH):
    delta = max_length - len(seq)
    seq = seq + [PAD for _ in range(delta)]
    # seq = [PAD for _ in range(delta - delta//2)] + seq + [PAD for _ in range(delta//2)]
    return np.asarray(seq)


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
    lrate = max(0.0001, initial_lrate * math.pow(drop, math.floor(epoch / epochs_drop)))
    return lrate


def batch_generator(stream, onto, classes):

    s_cls = set(classes)

    def prepare(batch):
        ids, X, labels = zip(*batch)
        b = max(map(len, X)) + 100
        X = [pad_seq(seq, b) for seq in X]
        Y = []
        for lbl in labels:
            y = np.zeros(len(classes))
            for go in onto.propagate(lbl, include_root=False):
                if go not in s_cls:
                    continue
                y[classes.index(go)] = 1
            Y.append(y)
        return ids, np.asarray(X), np.asarray(Y)

    data = sorted(stream, key=lambda elem: elem[0])

    batch = []
    for k, x, y in data:
        batch.append([k, x, y])
        if len(batch) == BATCH_SIZE:
            yield prepare(batch)
            batch = []
    yield prepare(batch)


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


def ModelCNN(classes):
    inp = Input(shape=(None,))
    out = Classifier(Features(inp), classes)
    model = Model(inputs=[inp], outputs=[out])
    adam = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='binary_crossentropy', optimizer=adam)

    return model


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB"),
    parser.add_argument("--aspect", type=str, choices=['F', 'P', 'C'],
                        default="F", help="Specify the ontology aspect.")
    parser.add_argument("--init_epoch", type=int, default=0,
                        help="Which epoch to start training the model?")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="How many epochs to train the model?")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train(model, gen_xy, length_xy, epoch, num_epochs,
          history=LossHistory(),
          lrate=LearningRateScheduler(step_decay)):

    pbar = tqdm(total=length_xy)

    for _, X, Y in gen_xy:
        assert len(X) == len(Y)

        model.fit(x=X, y=Y,
                  batch_size=BATCH_SIZE,
                  epochs=epoch + 1,
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


def calc_loss(y_true, y_pred, batch_size=BATCH_SIZE):
    return np.mean([log_loss(y, y_hat) for y, y_hat in zip(y_true, y_pred)])


def predict(model, gen_xy, length_xy, classes):
    pbar = tqdm(total=length_xy, desc="Predicting...")
    i, m, n = 0, length_xy, len(classes)
    y_pred, y_true = np.zeros((m, n)), np.zeros((m, n))
    ids = list()

    for i, (uid, X, Y) in enumerate(gen_xy):
        assert len(X) == len(Y)
        k = len(Y)
        y_hat, y = model.predict(X), Y
        y_pred[i:i + k, ], y_true[i:i + k, ] = y_hat, y
        ids.append(uid)
        pbar.update(k)
    pbar.close()
    return ids, y_true, y_pred


def evaluate(y_true, y_pred, classes):
    y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
    y_true = y_true[~np.all(y_true == 0, axis=1)]
    prs, rcs, f1s = performance(y_pred, y_true, classes)
    return calc_loss(y_true, y_pred), prs, rcs, f1s


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    ASPECT = args.aspect  # default: Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(ASPECT)

    classes = get_classes(db, onto)
    classes.remove(onto.root)
    assert onto.root not in classes

    model = ModelCNN(classes)
    if args.resume:
        model.load_weights(args.resume)
        print("Loaded model from disk")
    model.summary()

    for epoch in range(args.init_epoch, args.num_epochs):

        trn_stream, tst_stream = get_training_and_validation_streams(db, onto, classes)

        train(model, batch_generator(trn_stream, onto, classes), len(trn_stream), epoch, args.num_epochs)
        _, y_true, y_pred = predict(model, batch_generator(tst_stream, onto, classes), len(tst_stream), classes)
        loss, prs, rcs, f1s = evaluate(y_true, y_pred, classes)
        i = np.argmax(f1s)

        print("[Epoch %d/%d] (Validation Loss: %.5f, F_max: %.3f, precision: %.3f, recall: %.3f)"
              % (epoch + 1, args.num_epochs, loss, f1s[i], prs[i], rcs[i]))

        if f1s[i] < 0.5: continue

        model_str = '%s-%d-%.5f-%.2f' % ("deeperseq", epoch + 1, loss, f1s[i])
        model.save_weights("checkpoints/%s.hdf5" % model_str)
        with open("checkpoints/%s.json" % model_str, "w+") as f:
            f.write(model.to_json())
        np.save("checkpoints/%s.npy" % model_str, np.asarray(classes))
