import os
import sys
import wget

import numpy as np

from numpy import unique
from Bio.SeqIO import parse as parse_fasta

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit


exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"]


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
        return "Biological_Process" if aspect == 'P' \
            else "Molecular_Function" if aspect == 'F' \
            else "Cellular_Component" if aspect == 'C' \
            else "unspecified" if not aspect \
            else "unknown"


# class Uniprot(object):
#
#     def __init__(self, doc):
#         self.uid = doc["_id"]
#         self.db_id = doc["db"]
#         self.name = doc["entry_name"]
#         self.seq = doc["sequence"]
#
#     @property
#     def name(self):
#         return self.__name
#
#     @name.setter
#     def name(self, val):
#         self.__name = val
#
#     @property
#     def seq(self):
#         return self.__seq
#
#     @seq.setter
#     def seq(self, val):
#         self.__seq = val
#
#     @property
#     def uid(self):
#         return self.__uid
#
#     @uid.setter
#     def uid(self, val):
#         self.__uid = val
#
#     def get_go_terms(self):
#         return set(
#             map(lambda e: e["GO_ID"], db.goa_uniprot.find({"DB_Object_ID": self.uid}))
#         )


class SequenceLoader(object):
    def __init__(self, src_sequence, num_sequences, source_name='<?>'):
        self.sequence_source = src_sequence
        self.num_sequences = num_sequences
        self.source_name = source_name

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

    def __init__(self, uid2seq, seq2vec, uid2lbl=None, lbl2vec=None, transform=Identity()):

        self.lbl2vec = lbl2vec
        self.seq2vec = seq2vec
        self.transform = transform
        self.records = records = []

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
        records.sort(key=lambda r: -len(r.seq))

        if uid2lbl:
            labels = list(record.lbl for record in records)
            self.lbl2vec = lbl2vec if lbl2vec else \
                MultiLabelBinarizer(sparse_output=False).fit(labels)

    @property
    def classes(self):
        return self.lbl2vec.classes_ if self.lbl2vec else None

    @property
    def batch_size(self):
        return self.lbl2vec.classes_ if self.lbl2vec else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        seq2vec = self.seq2vec
        lbl2vec = self.lbl2vec
        record = self.records[i]
        f = self.transform
        return f(seq2vec[record.seq]), lbl2vec.transform([record.lbl]) \
            if lbl2vec else f(seq2vec[record.seq])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DataLoader(object):

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):

        dataset = self.dataset
        classes = dataset.classes
        batch_size = self.batch_size

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
                                        cutoff_date, aspect, exp=True):

    query = {"DB": "UniProtKB", "Date": {"$lte": cutoff_date}}

    if exp:
        query["Evidence"] = {"$in": exp_codes}
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


def rm_if_less_than(m, direct_dict, reverse_dict):
    labels_to_be_deleted = set()
    for uid in reverse_dict.keys():
        if len(reverse_dict[uid]) < m:
            labels_to_be_deleted.add(uid)

    uids_to_be_deleted = set()
    for uid in direct_dict:
        direct_dict[uid] -= labels_to_be_deleted
        if len(direct_dict[uid]) == 0:
            uids_to_be_deleted.add(uid)

    for uid in uids_to_be_deleted:
        del direct_dict[uid]

    return direct_dict, reverse_dict


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


def load_target_files(datadir):
    pass


def load_benchmark_files(datadir):
    pass


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
