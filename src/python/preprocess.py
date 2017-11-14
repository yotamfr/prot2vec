import os
import sys
import wget
import datetime
import numpy as np

from numpy import unique
from Bio.SeqIO import parse as parse_fasta

from sklearn.preprocessing import MultiLabelBinarizer


exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"]


cafa2_cutoff = datetime.datetime(2014, 1, 1, 0, 0)
cafa3_cutoff = datetime.datetime(2017, 2, 2, 0, 0)
today_cutoff = datetime.datetime.now()

cafa3_targets_url = 'http://biofunctionprediction.org/cafa-targets/CAFA3_targets.tgz'
cafa3_train_url = 'http://biofunctionprediction.org/cafa-targets/CAFA3_training_data.tgz'
cafa2_data_url = 'https://ndownloader.figshare.com/files/3658395'
cafa2_targets_url = 'http://biofunctionprediction.org/cafa-targets/CAFA-2013-targets.tgz'


def blocks(files, size=8192*1024):
    while True:
        buffer = files.read(size)
        if not buffer:
            break
        yield buffer


def count_lines(fpath, sep=bytes('\n', 'utf8')):
    with open(fpath, "rb") as f:
        return sum(bl.count(sep) for bl in blocks(f))


class GoAspect(object):
    def __init__(self, aspect=None):
        self._aspect = aspect

    @property
    def aspect(self):
        return self._aspect if self._aspect else 'unspecified'

    def __eq__(self, other):
        return (str(self) == str(other)) or (self.aspect == other)

    def __str__(self):
        aspect = self._aspect
        return "BPO" if aspect == 'P' \
            else "MFO" if aspect == 'F' \
            else "CCO" if aspect == 'C' \
            else "unspecified" if not aspect \
            else "unknown"


class SequenceLoader(object):
    def __init__(self, src_sequence, num_sequences, source_name):
        self.sequence_source = src_sequence
        self.num_sequences = num_sequences
        self.source_name = source_name if source_name else '<?>'

    def load(self):
        n = self.num_sequences if self.num_sequences else '<?>'
        print("Loading: \"%s\"" % self.source_name)

        seq_id2seq = dict()
        for i, seq in enumerate(self.sequence_source):
            sys.stdout.write("\r{0:.0f}%".format(100.0 * i/n))
            seq_id, seq_seq = self.parse_sequence(seq)
            seq_id2seq[seq_id] = seq_seq

        print("\nFinished loading %s sequences!" % len(seq_id2seq))

        return seq_id2seq

    def parse_sequence(self, seq):
        return None, None


class FastaFileLoader(SequenceLoader):
    def __init__(self, src_fasta, num_seqs, filename):
        super(FastaFileLoader, self).__init__(src_fasta, num_seqs, filename)

    def parse_sequence(self, seq):
        return seq.id, seq.seq


class UniprotCollectionLoader(SequenceLoader):
    def __init__(self, src_sequence, num_sequences, src_name):
        super(UniprotCollectionLoader, self)\
            .__init__(src_sequence, num_sequences, src_name)

    def parse_sequence(self, doc):
        return doc["_id"], doc["sequence"]


class MappingLoader(object):
    def __init__(self, src_mapping, num_mapping, source_name='<?>'):
        self.mapping_source = src_mapping
        self.mapping_count = num_mapping
        self.source_name = source_name

    def load(self):

        n = self.mapping_count if self.mapping_count else '<?>'
        print("Loading: \"%s\"" % self.source_name)

        direct_map, reverse_map = dict(), dict()
        for i, item in enumerate(self.mapping_source):
            sys.stdout.write("\r{0:.0f}%".format(100.0 * i/n))
            id1, id2 = self.parse_mapping(item)
            if (not id1) and (not id2):
                continue
            try:
                if id1 in direct_map:
                    direct_map[id1].add(id2)
                else:
                    direct_map[id1] = {id2}
                if id2 in reverse_map:
                    reverse_map[id2].add(id1)
                else:
                    reverse_map[id2] = {id1}
            except TypeError:
                pass

        m = sum(map(len, direct_map.values()))
        print("\nFinished loading %s mappings!" % m)

        return direct_map, reverse_map

    def parse_mapping(self, entry):
        return None, None


class MappingFileLoader(MappingLoader):
    def __init__(self, file_src, line_num, filename):
        super(MappingFileLoader, self).__init__(file_src, line_num, filename)

    def parse_mapping(self, line):
        s_line = line.strip().split()
        if len(s_line) != 2:
            return None, None
        else:
            return s_line


class GoAnnotationLoader(MappingLoader):
    def __init__(self, src_annotations, num_annotations, src_name, aspect=GoAspect()):
        super(GoAnnotationLoader, self)\
            .__init__(src_annotations, num_annotations, src_name)
        self.aspect = aspect

    def parse_mapping(self, entry):
        return None, None


class GoAnnotationFileLoader(GoAnnotationLoader):
    def __init__(self, annotation_file_io, num_lines, src_name, aspect):
        super(GoAnnotationFileLoader, self)\
            .__init__(annotation_file_io, num_lines, src_name, aspect)

    def parse_mapping(self, line):
        seq_id, go_id, go_asp = line.strip().split('\t')
        if go_asp == self.aspect:
            return seq_id, go_id
        else:
            return None, None


class GoAnnotationCollectionLoader(GoAnnotationLoader):
    def __init__(self, annotation_cursor, annotation_count, src_name, aspect):
        super(GoAnnotationCollectionLoader, self) \
            .__init__(annotation_cursor, annotation_count, src_name, aspect)

    def parse_mapping(self, doc):
        seq_id, go_id, go_asp = doc["DB_Object_ID"], doc["GO_ID"], doc["Aspect"]
        if go_asp == self.aspect:
            return seq_id, go_id
        else:
            return None, None


class Record(object):
    pass


class Seq2Vec(object):

    def __init__(self, model):
        self._w2v = model

    def __getitem__(self, seq):
        return np.array([self._w2v[aa] for aa in seq], dtype=np.float64)


class Identity(object):

    def __call__(self, x):
        return x


class Dataset(object):

    def __init__(self, uid2seq, emb, uid2lbl=None, mlb=None, transform=Identity()):

        self._mlb = mlb
        self._emb = emb
        self.transform = transform
        self.records = []
        self.do_init(uid2seq, uid2lbl)

    def do_init(self, uid2seq, uid2lbl):
        records = self.records = []
        keys = uid2seq.keys() \
            if not uid2lbl \
            else uid2lbl.keys()
        for uid in keys:
            record = Record()
            record.uid = uid
            record.seq = uid2seq[uid]
            if uid2lbl:
                record.lbl = uid2lbl[uid]
            records.append(record)
            
    @staticmethod
    def to_dictionaries(records):
        uid2seq = {record.uid: record.seq for record in records}
        uid2lbl = {record.uid: record.lbl for record in records}
        return uid2seq, uid2lbl

    def update(self, other):
        uid2seq, uid2lbl = Dataset.to_dictionaries(self.records)
        for record in other.records:
            if record.uid in uid2lbl:
                uid2lbl[record.uid] |= record.lbl
            else:
                uid2lbl[record.uid] = record.lbl
        uid2seq.update(Dataset.to_dictionaries(other.records)[0])
        self.do_init(uid2seq, uid2lbl)
        return self

    def split(self, ratio=0.2):     # split and make sure distrib. of length is the same
        data = np.array(sorted(self.records, key=lambda r: len(r.seq)))
        n, mlb = len(data), self.mlb
        train_indx = np.array([bool(i % round(1/ratio)) for i in range(n)])
        test_indx = np.invert(train_indx)
        train_records, test_records = data[train_indx], data[test_indx]
        train_uid2sec, train_uid2lbl = Dataset.to_dictionaries(train_records)
        test_uid2sec, test_uid2lbl = Dataset.to_dictionaries(test_records)
        return Dataset(train_uid2sec, self._emb, train_uid2lbl, mlb,
                       transform=self.transform),\
               Dataset(test_uid2sec, self._emb, test_uid2lbl, mlb,
                       transform=self.transform)

    @property
    def mlb(self):
        return self._mlb if self._mlb \
            else MultiLabelBinarizer(sparse_output=True).fit(self.labels)

    @mlb.setter
    def mlb(self, mlb):
        self._mlb = mlb

    @property
    def labels(self):
        return list(record.lbl for record in self.records)

    @property
    def classes(self):
        return self._mlb.classes_ if self._mlb else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        emb, mlb, fn = self._emb, self._mlb, self.transform
        record = self.records[i]
        return fn(emb[record.seq]), mlb.transform([record.lbl]) \
            if mlb else fn(emb[record.seq])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DataLoader(object):

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):

        dataset = self.dataset
        classes = dataset.mlb.classes_
        batch_size = self.batch_size

        dataset.records.sort(key=lambda r: -len(r.seq))

        n = len(dataset)

        seq, lbl = dataset[0]
        B = min(n, batch_size)
        T, D = seq.shape

        batch_lbl = np.zeros((B, len(classes)))
        batch_seq = np.zeros((B, 1, D, T))

        i = 0
        while i < n:

            j = 0
            while j < B:
                seq, lbl = dataset[i + j]
                B = min(n - i, batch_size)
                L, D = seq.shape
                batch_seq[j, :, :, :L] = seq.reshape((D, L))
                batch_lbl[j, :] = lbl.reshape((len(classes),))
                j += 1

            i += j
            yield batch_seq[:B, :, :, :T], batch_lbl[:B, :]


def load_training_data_from_collections(annot_collection, seq_collection,
                                        from_date, to_date, aspect, exp=True, names=None):
    query = {"DB": "UniProtKB"}
    if from_date and to_date:
        query["Date"] = {"$gte": from_date, "$lte": to_date}
    elif to_date:
        query["Date"] = {"$lte": to_date}
    elif from_date:
        query["Date"] = {"$gte": from_date}
    if exp:
        query["Evidence"] = {"$in": exp_codes}
    if names:
        query["DB_Object_Symbol"] = {"$in": names}

    annot_src = annot_collection.find(query)
    annot_num = annot_collection.count(query)
    seq_id2go_id, go_id2seq_id = \
        GoAnnotationCollectionLoader(annot_src, annot_num,
                                     annot_collection, aspect).load()

    query = {"_id": {"$in": unique(list(seq_id2go_id.keys())).tolist()}}
    sequence_src = seq_collection.find(query)
    sequence_num = seq_collection.count(query)
    seq_id2seq = UniprotCollectionLoader(sequence_src, sequence_num,
                                         seq_collection).load()

    return seq_id2seq, seq_id2go_id, go_id2seq_id


def filter_labels_by(filter_func, direct_dict, reverse_dict):
    labels_to_be_deleted = set()
    for uid in reverse_dict.keys():
        if filter_func(reverse_dict[uid]):
            labels_to_be_deleted.add(uid)
    sequences_to_be_deleted = set()
    for uid in direct_dict:
        direct_dict[uid] -= labels_to_be_deleted
        if len(direct_dict[uid]) == 0:
            sequences_to_be_deleted.add(uid)
    for uid in sequences_to_be_deleted:
        del direct_dict[uid]


def filter_sequences_by(filter_func, seq_dict, lbl_dict):
    uids_to_be_deleted = set()
    for uid in seq_dict:
        if filter_func(seq_dict[uid]):
            uids_to_be_deleted.add(uid)
    for uid in uids_to_be_deleted:
        if uid in lbl_dict:
            del lbl_dict[uid]
        del seq_dict[uid]


def load_training_data_from_files(annots_tsv, fasta_fname, aspect):
    annot_src = open(annots_tsv, 'r')
    num_annot = count_lines(annots_tsv, sep=bytes('\n', 'utf8'))
    seq_id2go_id, go_id2seq_id = \
        GoAnnotationFileLoader(annot_src, num_annot, annots_tsv, aspect).load()

    num_seq = count_lines(fasta_fname, sep=bytes('>', 'utf8'))
    fasta_src = parse_fasta(open(fasta_fname, 'r'), 'fasta')
    seq_id2seq = FastaFileLoader(fasta_src, num_seq, fasta_fname).load()

    return seq_id2seq, seq_id2go_id, go_id2seq_id


def load_cafa3_targets(targets_dir, mapping_dir):

    trg_id2seq, trg_id2seq_id, seq_id2trg_id = dict(), dict(), dict()

    for fname in os.listdir(targets_dir):
        fpath = "%s/%s" % (targets_dir, fname)
        num_seq = count_lines(fpath, sep=bytes('>', 'utf8'))
        fasta_src = parse_fasta(open(fpath, 'r'), 'fasta')
        trg_id2seq.update(FastaFileLoader(fasta_src, num_seq, fname).load())

    for fname in os.listdir(mapping_dir):
        fpath = "%s/%s" % (mapping_dir, fname)
        num_mapping = count_lines(fpath, sep=bytes('\n', 'utf8'))
        src_mapping = open(fpath, 'r')
        d1, d2 = MappingFileLoader(src_mapping, num_mapping, fname).load()
        trg_id2seq_id.update(d1)
        seq_id2trg_id.update(d2)

    return trg_id2seq, trg_id2seq_id, seq_id2trg_id


def unzip(src, trg):
    if ".zip" in src:
        res = os.system('unzip %s -d %s' % (src, trg))
        assert res == 0
    elif ".tgz" in src:
        res = os.system('tar -xvzf %s -C %s' % (src, trg))
        assert res == 0
    else:
        res = os.system('unzip %s -d %s' % (src, trg))
        if res != 0: print("failed to decompress")


def wget_and_unzip(sub_dir, rel_dir, url):
    print("Downloading %s" % sub_dir)
    fname = wget.download(url, out=rel_dir)
    unzip(fname, rel_dir)
    os.remove(fname)


def load_cafa3(db, data_dir, asp, aa_emb, seqs_filter=None, lbls_filter=None, trans=None):

    aspect = GoAspect(asp)
    seq2vec = Seq2Vec(aa_emb)
    cafa3_train_dir = '%s/CAFA3_training_data' % data_dir
    if not os.path.exists(cafa3_train_dir):
        wget_and_unzip('CAFA3_training_data', data_dir, cafa3_train_url)

    cafa3_go_tsv = '%s/%s/uniprot_sprot_exp.txt' % (data_dir, cafa3_train_dir)
    cafa3_train_fasta = '%s/%s/uniprot_sprot_exp.fasta' % (data_dir, cafa3_train_dir)
    seq_id2seq, seq_id2go_id, go_id2seq_id = \
        load_training_data_from_files(cafa3_go_tsv, cafa3_train_fasta, asp)
    if seqs_filter:
        filter_sequences_by(seqs_filter, seq_id2seq, seq_id2go_id)
    if lbls_filter:
        filter_labels_by(lbls_filter, seq_id2go_id, go_id2seq_id)
    train_set = Dataset(seq_id2seq, seq2vec, seq_id2go_id, transform=trans)

    cafa3_targets_dir = '%s/Target files' % data_dir
    cafa3_mapping_dir = '%s/Mapping files' % data_dir
    if not os.path.exists(cafa3_targets_dir) or not os.path.exists(cafa3_mapping_dir):
        wget_and_unzip('CAFA3_targets', data_dir, cafa3_targets_url)

    annots_fname = 'leafonly_%s_unique.txt' % aspect
    annots_fpath = '%s/CAFA3_benchmark20170605/groundtruth/%s' % (data_dir, annots_fname)
    trg_id2seq, _, _ = load_cafa3_targets(cafa3_targets_dir, cafa3_mapping_dir)
    num_mapping = count_lines(annots_fpath, sep=bytes('\n', 'utf8'))
    src_mapping = open(annots_fpath, 'r')
    trg_id2go_id, go_id2trg_id = MappingFileLoader(src_mapping, num_mapping, annots_fname).load()
    if seqs_filter:
        filter_sequences_by(seqs_filter, trg_id2seq, trg_id2go_id)
    if lbls_filter:
        filter_labels_by(lbls_filter, trg_id2go_id, go_id2trg_id)
    test_set = Dataset(trg_id2seq, seq2vec, trg_id2go_id, transform=trans)

    seq_id2seq, seq_id2go_id, go_id2seq_id = \
        load_training_data_from_collections(db.goa_uniprot, db.uniprot,
                                            cafa3_cutoff, today_cutoff, asp)
    if seqs_filter:
        filter_sequences_by(seqs_filter, seq_id2seq, seq_id2go_id)
    if lbls_filter:
        filter_labels_by(lbls_filter, seq_id2go_id, go_id2seq_id)
    valid_set = Dataset(seq_id2seq, seq2vec, seq_id2go_id, transform=trans)

    all_labels = np.concatenate([train_set.labels, valid_set.labels, test_set.labels])
    mlb = MultiLabelBinarizer(sparse_output=False).fit(all_labels)
    train_set.mlb, valid_set.mlb, test_set.mlb = mlb, mlb, mlb

    return train_set, valid_set, test_set, mlb


def load_cafa2(db, data_dir, aa_emb, seqs_filter=None, lbls_filter=None, transform=None):

    sub_dir = cafa2_targets_dir = 'CAFA-2013-targets'
    if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
        wget_and_unzip(sub_dir, data_dir, cafa2_targets_url)
    sub_dir = cafa2_data_dir = 'CAFA2Supplementary_data'
    if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
        wget_and_unzip(sub_dir, data_dir, cafa2_data_url)

    cafa2_targets_dir = './CAFA2Supplementary_data/data/CAFA2-targets'
    cafa2_benchmark_dir = './CAFA2Supplementary_data/data/benchmark'
