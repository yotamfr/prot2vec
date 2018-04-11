import os
import sys

from Bio import SeqIO
from Bio.Seq import Seq as BioSeq
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO import parse as parse_fasta

from numpy import unique

from src.python.geneontology import *

from src.python.consts import *

MAX_LENGTH = 2000
MIN_LENGTH = 1

t0 = datetime(2014, 1, 1, 0, 0)
t1 = datetime(2014, 9, 1, 0, 0)

verbose = True

np.random.seed(1)


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


def propagate(seq2go, onto, include_root):
    seq2go_cpy = {}
    for i, (k, v) in enumerate(seq2go.items()):
        sys.stdout.write("\r{0:.0f}%".format(100.0 * i / len(seq2go)))
        seq2go_cpy[k] = onto.propagate(v, include_root=include_root)
    return seq2go_cpy


def get_classes(seq2go, onto=None):
    if onto: seq2go = propagate(seq2go, onto, include_root=False)
    return reduce(lambda x, y: set(x) | set(y), seq2go.values(), set())


def get_classes_trn_tst(onto, seq2go_trn, seq2go_tst, propagate=False):
    return onto.sort(get_classes(seq2go_trn, onto, propagate)
                     | get_classes(seq2go_tst, onto, propagate))


def get_random_training_and_validation_streams(db, asp, ratio, profile=False):
    onto = get_ontology(asp)
    Stream = ProfileStream if profile else SequenceStream
    collection = db.pssm if profile else db.uniprot
    q_valid = {'DB': 'UniProtKB', 'Evidence': {'$in': exp_codes}, 'Aspect': asp}
    seq2go, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid),
                                             db.goa_uniprot.count(q_valid), asp).load()
    seq2go_tst = {k: seq2go[k] for k in np.random.choice(list(seq2go.keys()), size=int(len(seq2go)*ratio))}
    seq2go_trn = {k: v for k, v in seq2go.items() if k not in seq2go_tst}
    stream_tst = Stream(seq2go_tst, collection, onto)
    stream_trn = Stream(seq2go_trn, collection, onto)
    return stream_trn, stream_tst


def load_training_and_validation(db, start, end, asp, limit=None):
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$lte": start},
               'Aspect': asp}

    sequences_train, annotations_train, _ = _get_labeled_data(db, q_train, asp, limit=None)

    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date':  {"$gt": start, "$lte": end},
               'Aspect': asp}

    sequences_valid, annotations_valid, _ = _get_labeled_data(db, q_valid, asp, limit=limit)
    forbidden = set(sequences_train.keys())
    sequences_valid = {k: v for k, v in sequences_valid.items() if k not in forbidden}
    annotations_valid = {k: v for k, v in annotations_valid.items() if k not in forbidden}

    return sequences_train, annotations_train, sequences_valid, annotations_valid


def _get_labeled_data(db, query, asp, limit=None, propagate=True):
    onto = get_ontology(asp)
    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    seqid2goid, goid2seqid = GoAnnotationCollectionLoader(s, c, asp).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    if propagate:
        for k, v in seqid2goid.items():
            annots = onto.propagate(v, include_root=False)
            seqid2goid[k] = annots

    return seqid2seq, seqid2goid, goid2seqid


def get_training_and_validation_streams(db, start, end, asp, profile=False):
    onto = get_ontology(asp)
    Stream = ProfileStream if profile else SequenceStream
    collection = db.pssm if profile else db.uniprot
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$lte": start},
               'Aspect': asp}
    seq2go_trn, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_train),
                                                 db.goa_uniprot.count(q_train), asp).load()
    stream_trn = Stream(seq2go_trn, collection, onto)
    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": start, "$lte": end},
               'Aspect': asp}
    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid),
                                                 db.goa_uniprot.count(q_valid), asp).load()
    seq2go_tst = {k: v for k, v in seq2go_tst.items() if k not in seq2go_trn}
    stream_tst = Stream(seq2go_tst, collection, onto)
    return stream_trn, stream_tst


def get_balanced_training_and_validation_streams(db, start, end, asp, onto, classes):
    collection = db.uniprot
    q_valid = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$gt": start, "$lte": end},
               'Aspect': asp}
    seq2go_tst, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_valid),
                                                 db.goa_uniprot.count(q_valid), asp).load()
    stream_tst = BinaryStream(seq2go_tst, collection, onto, classes)
    q_train = {'DB': 'UniProtKB',
               'Evidence': {'$in': exp_codes},
               'Date': {"$lte": start},
               'Aspect': asp}
    seq2go_trn, _ = GoAnnotationCollectionLoader(db.goa_uniprot.find(q_train),
                                                 db.goa_uniprot.count(q_train), asp).load()
    seq2go_trn = {k:v for k,v in seq2go_trn.items() if k not in seq2go_tst}
    stream_trn = BalancedBinaryStream(seq2go_trn, collection, onto, classes)
    return stream_trn, stream_tst


def pssm2matrix(seq, pssm):
    if pssm:
        return [AA.aa2onehot[aa] + [pssm[i][AA.index2aa[k]] for k in range(20)] for i, aa in enumerate(seq)]
    else:
        return []


class DataStream(object):
    def __init__(self, seq2go, collection, onto, limit=None):
        query = {"_id": {"$in": list(seq2go.keys())}}
        count = limit if limit else collection.count(query)
        source = collection.find(query).batch_size(10)
        if limit: source = source.limit(limit)
        self._count = count
        self._source = source
        self._seq2go = seq2go
        self._onto = onto

    def __len__(self):
        return self._count


class SequenceStream(DataStream):
    def __init__(self, seq2go, collection, onto):
        super(SequenceStream, self).__init__(seq2go, collection, onto)

    def __iter__(self):
        count = self._count
        source = self._source
        seq2go = self._seq2go
        cls = set(self._onto.classes)
        for unipid, seq in UniprotCollectionLoader(source, count):
            if unipid not in seq2go:
                continue
            if not np.all([go in cls for go in seq2go[unipid]]):
                continue
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue
            yield [unipid, seq, seq2go[unipid]]

    def to_dictionaries(self, propagate=False):
        go2uid = {}
        uid2seq = {}
        uid2go = {}
        onto = self._onto
        for unipid, seq, annots in self:
            if propagate:
                annots = onto.propagate(annots, include_root=False)
            for go in annots:
                if go in go2uid:
                    go2uid[go].append(unipid)
                else:
                    go2uid[go] = [unipid]
            uid2seq[unipid] = seq
            uid2go[seq] = annots
        return uid2seq, uid2go, go2uid

    def to_fasta(self, out_file):
        sequences = []
        for unipid, seq, annots in self:
            sequences.append(SeqRecord(BioSeq(seq), unipid))
        SeqIO.write(sequences, open(out_file, 'w+'), "fasta")


class ProfileStream(DataStream):
    def __init__(self, seq2go, collection, onto):
        super(ProfileStream, self).__init__(seq2go, collection, onto)

    def __iter__(self):
        count = self._count
        source = self._source
        seq2go = self._seq2go
        onto = self._onto
        for uid, (seq, pssm, aln) in PssmCollectionLoader(source, count):
            if uid not in seq2go:
                continue
            if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                continue
            mat = pssm2matrix(seq, pssm)
            yield [uid, seq, mat, seq2go[uid]]


class BinaryStream(DataStream):
    def __init__(self, seq2go, collection, onto, classes):
        super(BinaryStream, self).__init__(seq2go, collection)
        self._collection = collection
        source = self._source
        count = self._count
        self._onto = onto
        self._seq2go = seq2go
        self._classes = classes
        self._seq2seq = UniprotCollectionLoader(source, count).load()
        self._go2seq = go2seq = {}
        self._seq2go_original = {}
        for seqid, terms in seq2go.items():
            self._seq2go_original[seqid] = terms
            prop = onto.propagate(terms, include_root=False)
            for go in prop:
                if go in go2seq:
                    go2seq[go].append(seqid)
                else:
                    go2seq[go] = [seqid]
            seq2go[seqid] = prop

    def __iter__(self):
        onto = self._onto
        go2seq = self._go2seq
        seq2seq = self._seq2seq
        seq2go = self._seq2go_original
        classes = onto.classes
        for pos, sequences in go2seq.items():
            for seqid in sequences:
                yield seqid, seq2seq[seqid], classes.index(pos), 1
                neg = onto.negative_sample(seq2go[seqid])
                yield seqid, seq2seq[seqid], classes.index(neg), 0

    def __len__(self):
        return sum(map(len, self._go2seq.values()))


class BalancedBinaryStream(BinaryStream):

    def __init__(self, seq2go, collection, onto, classes):
        super(BalancedBinaryStream, self).__init__(seq2go, collection, onto, classes)

    def __iter__(self, ssz=512):  # ssz == sample_size
        onto = self._onto
        classes = self._classes
        go2seq = self._go2seq
        seq2seq = self._seq2seq
        seq2go = self._seq2go_original
        for i in range(1, onto.num_levels):
            for pos in onto.get_level(i):
                if pos not in go2seq:
                    continue
                for seqid in np.random.choice(go2seq[pos], max(ssz, 8)):
                    seq = seq2seq[seqid]
                    if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
                        continue
                    yield seqid, seq, classes.index(pos), 1
                    neg = onto.negative_sample(seq2go[seqid])
                    assert not onto.is_father(neg, seq2go[seqid])
                    yield seqid, seq, classes.index(neg), -1
            ssz //= 2

    def __len__(self):
        return -1


def load_data(stream, sort=True):
    data = []
    for i, packet in enumerate(stream):
        sys.stdout.write("\rLoading {0:.0f}".format(i))
        data.append(packet)
    if sort: data.sort(key=lambda p: len(p[1]), reverse=True)  # it is important to have seq @ 2nd place
    return data


if __name__ == "__main__":
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prot2vec']
    print("Indexing Data...")
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, 'F')
    print("Loading Data...")
    trn_stream.to_fasta('../../Data/training_set.fasta')
    tst_stream.to_fasta('../../Data/validation_set.fasta')
