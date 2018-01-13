import os
import sys
import wget
import datetime

import numpy as np

from numpy import unique

from Bio.SeqIO import parse as parse_fasta

from .geneontology import *

exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"] + ["TAS", "IC"]

cafa2_cutoff = datetime.datetime(2014, 1, 1, 0, 0)
cafa3_cutoff = datetime.datetime(2017, 2, 2, 0, 0)
today_cutoff = datetime.datetime.now()

cafa3_targets_url = 'http://biofunctionprediction.org/cafa-targets/CAFA3_targets.tgz'
cafa3_train_url = 'http://biofunctionprediction.org/cafa-targets/CAFA3_training_data.tgz'
cafa2_data_url = 'https://ndownloader.figshare.com/files/3658395'
cafa2_targets_url = 'http://biofunctionprediction.org/cafa-targets/CAFA-2013-targets.tgz'

verbose = True


class AminoAcids(object):

    def __init__(self):
        self.aa2index = \
            {
                "A": 0,
                "R": 1,
                "N": 2,
                "D": 3,
                "C": 4,
                "E": 5,
                "Q": 6,
                "G": 7,
                "H": 8,
                "I": 9,
                "L": 10,
                "K": 11,
                "M": 12,
                "F": 13,
                "P": 14,
                "S": 15,
                "T": 16,
                "W": 17,
                "Y": 18,
                "V": 19,
                "X": 20,
                "B": 21,
                "Z": 22,
                "O": 23,
                "U": 24
            }
        self.index2aa = {v: k for k, v in self.aa2index.items()}


AA = AminoAcids()


def blocks(files, size=8192*1024):
    while True:
        buffer = files.read(size)
        if not buffer:
            break
        yield buffer


def count_lines(fpath, sep=bytes('\n', 'utf8')):
    with open(fpath, "rb") as f:
        return sum(bl.count(sep) for bl in blocks(f))


class SequenceLoader(object):
    def __init__(self, src_sequence, num_sequences):
        self.sequence_source = src_sequence
        self.num_sequences = num_sequences

    def load(self):
        n = self.num_sequences
        seq_id2seq = dict()
        for i, seq in enumerate(self.sequence_source):
            if verbose:
                sys.stdout.write("\r{0:.0f}%".format(100.0 * i/n))
            seq_id, seq_seq = self.parse_sequence(seq)
            seq_id2seq[seq_id] = seq_seq

        if verbose:
            print("\nFinished loading %s sequences!" % len(seq_id2seq))

        return seq_id2seq

    def parse_sequence(self, seq):
        return None, None


class FastaFileLoader(SequenceLoader):
    def __init__(self, src_fasta, num_seqs):
        super(FastaFileLoader, self).__init__(src_fasta, num_seqs)

    def parse_sequence(self, seq):
        return seq.id, seq.seq


class UniprotCollectionLoader(SequenceLoader):
    def __init__(self, src_sequence, num_sequences):
        super(UniprotCollectionLoader, self).__init__(src_sequence, num_sequences)

    def parse_sequence(self, doc):
        return doc["_id"], doc["sequence"]


class MappingLoader(object):
    def __init__(self, src_mapping, num_mapping):
        self.mapping_source = src_mapping
        self.mapping_count = num_mapping

    def load(self):
        n = self.mapping_count
        direct_map, reverse_map = dict(), dict()
        for i, item in enumerate(self.mapping_source):
            if verbose:
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

        if verbose:
            m = sum(map(len, direct_map.values()))
            print("\nFinished loading %s mappings!" % m)

        return direct_map, reverse_map

    def parse_mapping(self, entry):
        return None, None


class MappingFileLoader(MappingLoader):
    def __init__(self, file_src, line_num):
        super(MappingFileLoader, self).__init__(file_src, line_num)

    def parse_mapping(self, line):
        s_line = line.strip().split()
        if len(s_line) != 2:
            return None, None
        else:
            return s_line


class GoAnnotationLoader(MappingLoader):
    def __init__(self, src_annotations, num_annotations, aspect=GoAspect()):
        super(GoAnnotationLoader, self)\
            .__init__(src_annotations, num_annotations)
        self.aspect = aspect

    def parse_mapping(self, entry):
        return None, None


class GoAnnotationFileLoader(GoAnnotationLoader):
    def __init__(self, annotation_file_io, num_lines, aspect):
        super(GoAnnotationFileLoader, self).__init__(annotation_file_io, num_lines, aspect)

    def parse_mapping(self, line):
        seq_id, go_id, go_asp = line.strip().split('\t')
        if go_asp == self.aspect:
            return seq_id, go_id
        else:
            return None, None


class GoAnnotationCollectionLoader(GoAnnotationLoader):
    def __init__(self, annotation_cursor, annotation_count, aspect):
        super(GoAnnotationCollectionLoader, self) .__init__(annotation_cursor, annotation_count, aspect)

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

    def __init__(self, uid2seq, uid2lbl, ontology,
                 embedder=Seq2Vec(AA.aa2index),
                 transform=Identity()):

        self._emb = embedder
        self.onto = ontology
        self.transform = transform
        self.records = []
        self.do_init(uid2seq, uid2lbl)
        self.augmented = False

    def do_init(self, uid2seq, uid2lbl):
        records = self.records = []
        keys = uid2lbl.keys()
        for uid in keys:
            record = Record()
            if uid not in uid2seq:
                continue
            record.uid = uid
            record.lbl = uid2lbl[uid]
            record.seq = uid2seq[uid]
            records.append(record)

    def augment(self, max_length=None):
        if self.augmented:
            return
        n, m = len(self), 0
        onto = self.onto
        for i, record in enumerate(self.records):
            if verbose:
                sys.stdout.write("\r{0:.0f}%".format(100.0 * i/n))
            record.lbl = onto.propagate(record.lbl, max_length)
        self.augmented = True

    def __str__(self):
        num_anno = sum(map(lambda record: len(record.lbl), self.records))
        num_go = len(reduce(lambda x, y: x | y, map(lambda r: set(r.lbl), self.records), set()))
        num_seq = len(self)
        s = '\n#Annotaions\t%d\n#GO-Terms\t%d\n#Sequences\t%d' % (num_anno, num_go, num_seq)
        return s

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
        n, onto = len(data), self.onto
        train_indx = np.array([bool(i % round(1/ratio)) for i in range(n)])
        test_indx = np.invert(train_indx)
        train_records, test_records = data[train_indx], data[test_indx]
        train_uid2sec, train_uid2lbl = Dataset.to_dictionaries(train_records)
        test_uid2sec, test_uid2lbl = Dataset.to_dictionaries(test_records)
        return Dataset(train_uid2sec, train_uid2lbl, onto, self._emb,
                       transform=self.transform),\
               Dataset(test_uid2sec, test_uid2lbl, onto, self._emb,
                       transform=self.transform)

    @property
    def labels(self):
        return list(record.lbl for record in self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        emb, onto, fn, r = self._emb, self.onto, self.transform, self.records[i]
        return fn(emb[r.seq]), np.array([onto[go] for go in r.lbl])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DataLoader(object):

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):

        dataset = self.dataset
        batch_size = self.batch_size

        dataset.records.sort(key=lambda r: -len(r.seq))

        M = len(dataset)

        seq, lbl = dataset[0]
        N = lbl.shape[1]

        B = min(M, batch_size)
        T, D = seq.shape

        batch_lbl = np.zeros((B, N))
        batch_seq = np.zeros((B, 1, D, T))

        i = 0
        while i < M:

            j = 0
            while j < B:
                seq, lbl = dataset[i + j]
                B = min(M - i, batch_size)
                L, D = seq.shape
                batch_seq[j, :, :, :L] = seq.reshape((D, L))
                batch_lbl[j, :] = lbl.reshape((N,))
                j += 1

            i += j
            yield batch_seq[:B, :, :, :T], batch_lbl[:B, :]


def load_training_data_from_collections(annot_collection, seq_collection, aspect,
                                        from_date=None, to_date=None, exp=True, names=None):
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
        GoAnnotationCollectionLoader(annot_src, annot_num, aspect).load()

    query = {"_id": {"$in": unique(list(seq_id2go_id.keys())).tolist()}}
    sequence_src = seq_collection.find(query)
    sequence_num = seq_collection.count(query)
    seq_id2seq = UniprotCollectionLoader(sequence_src, sequence_num).load()

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
        GoAnnotationFileLoader(annot_src, num_annot, aspect).load()

    num_seq = count_lines(fasta_fname, sep=bytes('>', 'utf8'))
    fasta_src = parse_fasta(open(fasta_fname, 'r'), 'fasta')
    seq_id2seq = FastaFileLoader(fasta_src, num_seq).load()

    return seq_id2seq, seq_id2go_id, go_id2seq_id


def load_cafa3_targets(targets_dir, mapping_dir):

    trg_id2seq, trg_id2seq_id, seq_id2trg_id = dict(), dict(), dict()

    for fname in os.listdir(targets_dir):
        print("\nLoading: %s" % fname)
        fpath = "%s/%s" % (targets_dir, fname)
        num_seq = count_lines(fpath, sep=bytes('>', 'utf8'))
        fasta_src = parse_fasta(open(fpath, 'r'), 'fasta')
        trg_id2seq.update(FastaFileLoader(fasta_src, num_seq).load())

    for fname in os.listdir(mapping_dir):
        print("\nLoading: %s" % fname)
        fpath = "%s/%s" % (mapping_dir, fname)
        num_mapping = count_lines(fpath, sep=bytes('\n', 'utf8'))
        src_mapping = open(fpath, 'r')
        d1, d2 = MappingFileLoader(src_mapping, num_mapping).load()
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


def load_data(db, asp='F', codes=exp_codes, limit=None):

    q = {'Evidence': {'$in': codes}, 'DB': 'UniProtKB'}
    c = limit if limit else db.goa_uniprot.count(q)
    s = db.goa_uniprot.find(q)
    if limit: s = s.limit(limit)

    seqid2goid, goid2seqid = GoAnnotationCollectionLoader(s, c, asp).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()
    onto = get_ontology(asp)

    return Dataset(seqid2seq, seqid2goid, onto), goid2seqid


def load_cafa3(db, data_dir, asp, aa_emb=AA.aa2index, seqs_filter=None, lbls_filter=None, trans=None):

    aspect = GoAspect(asp)
    seq2vec = Seq2Vec(aa_emb)
    cafa3_train_dir = '%s/CAFA3_training_data' % data_dir
    if not os.path.exists(cafa3_train_dir):
        wget_and_unzip('CAFA3_training_data', data_dir, cafa3_train_url)

    cafa3_go_tsv = '%s/uniprot_sprot_exp.txt' % cafa3_train_dir
    cafa3_train_fasta = '%s/uniprot_sprot_exp.fasta' % cafa3_train_dir
    seq_id2seq, seq_id2go_id, go_id2seq_id = \
        load_training_data_from_files(cafa3_go_tsv, cafa3_train_fasta, asp)
    if seqs_filter:
        filter_sequences_by(seqs_filter, seq_id2seq, seq_id2go_id)
    if lbls_filter:
        filter_labels_by(lbls_filter, seq_id2go_id, go_id2seq_id)
    train_set = Dataset(seq_id2seq, seq_id2go_id, seq2vec, transform=trans)

    cafa3_targets_dir = '%s/Target files' % data_dir
    cafa3_mapping_dir = '%s/Mapping files' % data_dir
    if not os.path.exists(cafa3_targets_dir) or not os.path.exists(cafa3_mapping_dir):
        wget_and_unzip('CAFA3_targets', data_dir, cafa3_targets_url)

    annots_fname = 'leafonly_%s_unique.txt' % aspect
    annots_fpath = '%s/CAFA3_benchmark20170605/groundtruth/%s' % (data_dir, annots_fname)
    trg_id2seq, _, _ = load_cafa3_targets(cafa3_targets_dir, cafa3_mapping_dir)
    num_mapping = count_lines(annots_fpath, sep=bytes('\n', 'utf8'))
    src_mapping = open(annots_fpath, 'r')
    trg_id2go_id, go_id2trg_id = MappingFileLoader(src_mapping, num_mapping).load()
    if seqs_filter:
        filter_sequences_by(seqs_filter, trg_id2seq, trg_id2go_id)
    if lbls_filter:
        filter_labels_by(lbls_filter, trg_id2go_id, go_id2trg_id)
    test_set = Dataset(trg_id2seq, trg_id2go_id, seq2vec, transform=trans)

    seq_id2seq, seq_id2go_id, go_id2seq_id = \
        load_training_data_from_collections(db.goa_uniprot, db.uniprot, asp,
                                            cafa3_cutoff, today_cutoff)
    if seqs_filter:
        filter_sequences_by(seqs_filter, seq_id2seq, seq_id2go_id)
    if lbls_filter:
        filter_labels_by(lbls_filter, seq_id2go_id, go_id2seq_id)
    valid_set = Dataset(seq_id2seq, seq_id2go_id, seq2vec, transform=trans)

    return train_set, valid_set, test_set


def load_cafa2(db, data_dir, aa_emb, seqs_filter=None, lbls_filter=None, transform=None):

    sub_dir = cafa2_targets_dir = 'CAFA-2013-targets'
    if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
        wget_and_unzip(sub_dir, data_dir, cafa2_targets_url)
    sub_dir = cafa2_data_dir = 'CAFA2Supplementary_data'
    if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
        wget_and_unzip(sub_dir, data_dir, cafa2_data_url)

    cafa2_targets_dir = './CAFA2Supplementary_data/data/CAFA2-targets'
    cafa2_benchmark_dir = './CAFA2Supplementary_data/data/benchmark'


if __name__ == "__main__":
    pass