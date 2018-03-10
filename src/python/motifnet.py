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
from keras.layers import MaxPooling2D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Concatenate, Flatten, Reshape
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LambdaCallback, LearningRateScheduler
# from keras.losses import hinge, binary_crossentropy
from keras import backend as K

from sklearn.metrics import log_loss

import math

import argparse

sess = tf.Session()
K.set_session(sess)

LR = 0.01
EPOCH_DROP = 1.0

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
    stream_trn = DataStream(source, count, seq2go_trn, onto, classes)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": start, "$lte": end},
               'Aspect': ASPECT}

    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid), db.goa_uniprot.count(q_valid), ASPECT).load()
    query = {"_id": {"$in": unique(list(seq2go_tst.keys())).tolist()}}
    count = limit if limit else db.uniprot.count(query)
    source = db.uniprot.find(query).batch_size(10)
    if limit: source = source.limit(limit)
    stream_tst = DataStream(source, count, seq2go_tst, onto, classes)

    return stream_trn, stream_tst


def pad_seq(seq, max_length=MAX_LENGTH):
    delta = max_length - len(seq)
    seq = [PAD for _ in range(delta - delta//2)] + seq + [PAD for _ in range(delta//2)]
    return np.asarray(seq)


class DataStream(object):
    def __init__(self, source, count, seq2go, onto, classes):

        self._classes = classes
        self._count = count
        self._source = source
        self._seq2go = seq2go
        self._onto = onto

    def __iter__(self):

        classes = self._classes
        count = self._count
        source = self._source
        seq2go = self._seq2go
        onto = self._onto

        s_cls = set(classes)

        for k, seq in UniprotCollectionLoader(source, count):
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue
            y = np.zeros(len(classes))
            for go in onto.propagate(seq2go[k], include_root=False):
                if go not in s_cls:
                    continue
                y[classes.index(go)] = 1

                x = [AA.aa2index[aa] for aa in seq]

            yield k, x, y

    def __len__(self):
        return self._count


def step_decay(epoch):
    initial_lrate = LR
    drop = 0.5
    epochs_drop = EPOCH_DROP
    lrate = max(0.0001, initial_lrate * math.pow(drop, math.floor(epoch / epochs_drop)))
    # print("lrate <- %.4f" % lrate)
    return lrate


def Motifs(inpt, filter_size=15, num_channels=250):
    motifs = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inpt)
    motifs = Conv1D(num_channels, filter_size, activation='relu', padding='valid')(motifs)
    motifs = Dropout(0.3)(motifs)
    return motifs


def Features(motifs, filter_size=5, dilation=1, num_channels=100):
    feats = Conv1D(num_channels, filter_size, dilation_rate=dilation, activation='relu')(motifs)
    feats = Dropout(0.3)(feats)
    return feats


def Classifier(inp1d, classes):
    out = Dense(len(classes))(inp1d)
    out = BatchNormalization()(out)
    out = Activation('sigmoid')(out)
    return out


def LargeInception(inpt, num_channels=64):

    tower_0 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_0 = Conv1D(num_channels, 3, padding='same', activation='relu')(tower_0)

    tower_1 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_1 = Conv1D(num_channels, 6, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_2 = Conv1D(num_channels, 12, padding='same', activation='relu')(tower_2)

    tower_3 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_3 = Conv1D(num_channels, 24, padding='same', activation='relu')(tower_3)

    return Concatenate(axis=2)([tower_0, tower_1, tower_2, tower_3])


def SmallInception(inpt, num_channels=64):

    tower_1 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_1 = Conv1D(num_channels, 6, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt)
    tower_2 = Conv1D(num_channels, 10, padding='same', activation='relu')(tower_2)

    return Concatenate(axis=2)([tower_1, tower_2])


def ResidualInception(inpt1, inpt2, num_channels=64):

    tower_11 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt1)
    tower_11 = Conv1D(num_channels, 6, padding='same', activation='relu')(tower_11)

    tower_12 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt1)
    tower_12 = Conv1D(num_channels, 10, padding='same', activation='relu')(tower_12)

    tower_21 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt2)
    tower_21 = Conv1D(num_channels, 6, padding='same', activation='relu')(tower_21)

    tower_22 = Conv1D(num_channels, 1, padding='same', activation='relu')(inpt2)
    tower_22 = Conv1D(num_channels, 10, padding='same', activation='relu')(tower_22)

    return Concatenate(axis=2)([tower_11, tower_12, tower_21, tower_22])


def ResInception(classes, opt):
    inpt = Input(shape=(None,))
    emb = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inpt)
    feats = LargeInception(LargeInception(emb))
    out = Classifier(GlobalMaxPooling1D()(feats), classes)
    model = Model(inputs=[inpt], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def DeepSeq(classes, opt):
    inpt = Input(shape=(None,))
    feats = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inpt)
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    out = Classifier(GlobalMaxPooling1D()(feats), classes)
    model = Model(inputs=[inpt], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def DeepeseqModule(inpt):
    feats = inpt
    feats = Conv1D(250, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    feats = Conv1D(100, 15, activation='relu', padding='valid')(feats)
    feats = Dropout(0.3)(feats)
    return feats


def MotifNet(classes, opt):
    inpt = Input(shape=(None,))
    emb = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inpt)
    motifnet = GlobalMaxPooling1D()(SmallInception(SmallInception(DeepeseqModule(emb))))
    out = Classifier(motifnet, classes)
    model = Model(inputs=[inpt], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def ProteinInception(classes, opt):
    inpt = Input(shape=(None,))
    emb = Embedding(input_dim=26, output_dim=23, embeddings_initializer='uniform')(inpt)
    feats = SmallInception(SmallInception(emb))
    out = Classifier(GlobalMaxPooling1D()(feats), classes)
    model = Model(inputs=[inpt], outputs=[out])
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def batch_generator(stream):

    def prepare(batch):
        ids, X, Y = zip(*batch)
        b = max(map(len, X)) + 100
        X = [pad_seq(seq, b) for seq in X]
        return ids, np.asarray(X), np.asarray(Y)

    batch = []
    for k, x, y in stream:
        if len(batch) == BATCH_SIZE:
            yield prepare(batch)
            batch = []
        batch.append([k, x, y])

    yield prepare(batch)


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB"),
    parser.add_argument("--aspect", type=str, choices=['F', 'P', 'C'],
                        default="F", help="Specify the ontology aspect.")
    parser.add_argument("--arch", type=str, choices=['motifnet', 'deepseq', 'inception', 'resinception'],
                        default="deepseq", help="Specify the model arch.")
    parser.add_argument("--opt", type=str, choices=['sgd', 'adam'],
                        default="adam", help="Specify the model optimizer.")
    parser.add_argument("--init_epoch", type=int, default=0,
                        help="Which epoch to start training the model?")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="How many epochs to train the model?")
    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')


class LossHistory(Callback):
    def __init__(self):
        self.losses = []
        self.max_size = 200

    def on_batch_end(self, batch, logs={}):
        self.losses.insert(0, logs.get('loss'))
        # while len(self.losses) > self.max_size:
        #     self.losses.pop()


def train(model, gen_xy, length_xy, epoch, num_epochs,
          history=LossHistory(), lrate=LearningRateScheduler(step_decay)):

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
    # return batch_size * np.mean([log_loss(y, y_hat) for y, y_hat in zip(y_true, y_pred)])
    return np.mean([log_loss(y, y_hat) for y, y_hat in zip(y_true, y_pred)])
    # return log_loss(y_true, y_pred, normalize=True)


def predict(model, gen_xy, length_xy, classes):
    pbar = tqdm(total=length_xy, desc="Predicting...")
    i, m, n = 0, length_xy, len(classes)
    y_pred, y_true = np.zeros((m, n)), np.zeros((m, n))
    ids = list()

    for i, (uid, X, Y) in enumerate(gen_xy):
        assert len(X) == len(Y)
        k = len(Y)
        ids.append(uid)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    ASPECT = args.aspect  # default: Molecular Function

    client = MongoClient(args.mongo_url)

    db = client['prot2vec']

    print("Loading Ontology...")
    onto = get_ontology(ASPECT)

    print("Listing Classes...")
    classes = get_classes(db, onto)
    classes.remove(onto.root)
    assert onto.root not in classes

    if args.opt == "adam":
        LR, EPOCH_DROP = 0.01, 1.0
        opt = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    else:
        LR, EPOCH_DROP = 1.0, 40.0
        opt = optimizers.SGD(lr=LR, momentum=0.9, nesterov=True)

    if args.arch == "motifnet":
        model = MotifNet(classes, opt)
    elif args.arch == "deepseq":
        model = DeepSeq(classes, opt)
    elif args.arch == "inception":
        model = ProteinInception(classes, opt)
    elif args.arch == "resinception":
        model = ResInception(classes, opt)
    else:
        print("Unknown model arch")
        exit(0)

    if args.resume:
        model.load_weights(args.resume)
        print("Loaded model from disk")

    model.summary()

    for epoch in range(args.init_epoch, args.num_epochs):

        trn_stream, tst_stream = get_training_and_validation_streams(db, onto, classes)

        train(model, batch_generator(trn_stream), len(trn_stream), epoch, args.num_epochs)
        _, y_true, y_pred = predict(model, batch_generator(tst_stream), len(tst_stream), classes)
        loss, prs, rcs, f1s = evaluate(y_true, y_pred, classes)
        i = np.argmax(f1s)

        print("[Epoch %d] (Validation Loss: %.5f, F_max: %.3f, precision: %.3f, recall: %.3f)"
              % (epoch + 1, loss, f1s[i], prs[i], rcs[i]))

        if f1s[i] < 0.5: continue

        model_str = '%s-%d-%.5f-%.2f' % (args.arch, epoch + 1, loss, f1s[i])
        model.save_weights("checkpoints/%s.hdf5" % model_str)
        with open("checkpoints/%s.json" % model_str, "w+") as f:
            f.write(model.to_json())
        np.save("checkpoints/%s.npy" % model_str, np.asarray(classes))
