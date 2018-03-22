import os
import sys
import wget

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from numpy import unique

from src.python.geneontology import *

from src.python.consts import *

MAX_LENGTH = 2800
MIN_LENGTH = 30

t0 = datetime(2014, 1, 1, 0, 0)
t1 = datetime(2014, 9, 1, 0, 0)

verbose = True


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

    def __iter__(self):
        for _, seq in enumerate(self.sequence_source):
            seq_id, seq_seq = self.parse_sequence(seq)
            if seq_seq is None or seq_id is None:
                continue
            yield seq_id, seq_seq

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


class PssmCollectionLoader(SequenceLoader):
    def __init__(self, src_sequence, num_sequences):
        super(PssmCollectionLoader, self).__init__(src_sequence, num_sequences)

    def parse_sequence(self, doc):
        if "seq" in doc and "pssm" in doc and "alignment" in doc:
            return doc["_id"], (doc["seq"], doc["pssm"], doc["alignment"])
        else:
            return None, None


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


def get_classes(db, onto, asp, start=t0, end=t1):

    q1 = {'DB': 'UniProtKB',
         'Evidence': {'$in': exp_codes},
         'Date': {"$lte": start},
         'Aspect': asp}
    q2 = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": start, "$lte": end},
               'Aspect': asp}

    def helper(q):
        seq2go, _ = GoAnnotationCollectionLoader(
            db.goa_uniprot.find(q), db.goa_uniprot.count(q), asp).load()
        for i, (k, v) in enumerate(seq2go.items()):
            sys.stdout.write("\r{0:.0f}%".format(100.0 * i / len(seq2go)))
            seq2go[k] = onto.propagate(v)
        return reduce(lambda x, y: set(x) | set(y), seq2go.values(), set())

    return onto.sort(helper(q1) | helper(q2))


def get_training_and_validation_streams(db, start, end, asp, profile=1, limit=None):

    if profile == 1:
        collection = db.pssm
        DataStream = ProfileStream
    else:
        collection = db.uniprot
        DataStream = SequenceStream

    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$lte": start},
               'Aspect': asp}
    seq2go_trn, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_train),
                                                 db.goa_uniprot.count(q_train), asp).load()
    query = {"_id": {"$in": unique(list(seq2go_trn.keys())).tolist()}}
    count = limit if limit else collection.count(query)
    source = collection.find(query).batch_size(10)
    if limit: source = source.limit(limit)
    stream_trn = DataStream(source, count, seq2go_trn)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": start, "$lte": end},
               'Aspect': asp}
    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid),
                                                 db.goa_uniprot.count(q_valid), asp).load()
    query = {"_id": {"$in": unique(list(seq2go_tst.keys())).tolist()}}
    count = limit if limit else collection.count(query)
    source = collection.find(query).batch_size(10)
    if limit: source = source.limit(limit)
    stream_tst = DataStream(source, count, seq2go_tst)

    return stream_trn, stream_tst


def pssm2matrix(seq, pssm):
    if pssm:
        return [AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)] for i, aa in enumerate(seq)]
    else:
        return []


class DataStream(object):
    def __init__(self, source, count, seq2go):
        self._count = count
        self._source = source
        self._seq2go = seq2go

    def __len__(self):
        return self._count


class SequenceStream(DataStream):
    def __init__(self, source, count, seq2go):
        super(SequenceStream, self).__init__(source, count, seq2go)

    def __iter__(self):
        count = self._count
        source = self._source
        seq2go = self._seq2go
        for uid, seq in UniprotCollectionLoader(source, count):
            if uid not in seq2go:
                continue
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue
            indices = [AA.aa2index[aa] for aa in seq]
            yield uid, indices, seq2go[uid]

    def to_fasta(self, out_file):
        count = self._count
        source = self._source
        seq2go = self._seq2go
        sequences = []
        for unipid, seq in UniprotCollectionLoader(source, count):
            if unipid not in seq2go:
                continue
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue
            sequences.append(SeqRecord(Seq(seq), unipid))
        SeqIO.write(sequences, open(out_file, 'w+'), "fasta")


class ProfileStream(DataStream):
    def __init__(self, source, count, seq2go):
        super(ProfileStream, self).__init__(source, count, seq2go)

    def __iter__(self):
        count = self._count
        source = self._source
        seq2go = self._seq2go
        for uid, (seq, pssm, aln) in PssmCollectionLoader(source, count):
            if uid not in seq2go:
                continue
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue
            indices = [AA.aa2index[aa] for aa in seq]
            mat = pssm2matrix(seq, pssm)
            yield uid, indices, mat, seq2go[uid]


def load_data(stream, reverse=True):
    data = []
    for i, packet in enumerate(stream):
        sys.stdout.write("\r{0:.0f}%".format(100.0 * i / len(stream)))
        data.append(packet)
    data.sort(key=lambda p: len(p[1]), reverse=reverse)  # it is important to have seq @ 2nd place
    return data


if __name__ == "__main__":
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prot2vec']
    print("Indexing Data...")
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, 'F', profile=0)
    print("Loading Data...")
    trn_stream.to_fasta('../../Data/training_set.fasta')
    tst_stream.to_fasta('../../Data/validation_set.fasta')
