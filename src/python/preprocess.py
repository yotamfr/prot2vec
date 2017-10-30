import os
import sys
import wget
import datetime

from numpy import unique
from Bio.SeqIO import parse as parse_fasta
from pymongo import MongoClient

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True,
                    help="Supply working directory for the data")
parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                    help="Supply the URL of MongoDB")
args = parser.parse_args()

client = MongoClient(args.mongo_url)
db_name = 'prot2vec'
db = client[db_name]

cafa2_cutoff = datetime.datetime(2014, 1, 1, 0, 0)
cafa3_cutoff = datetime.datetime(2017, 2, 2, 0, 0)

exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"]

cafa3_targets_url = 'http://biofunctionprediction.org/cafa-targets/CAFA3_targets.tgz'
cafa3_train_url = 'http://biofunctionprediction.org/cafa-targets/CAFA3_training_data.tgz'
cafa2_data_url = 'https://ndownloader.figshare.com/files/3658395'
cafa2_targets_url = 'http://biofunctionprediction.org/cafa-targets/CAFA-2013-targets.tgz'

cafa3_benchmark_dir = './CAFA3_benchmark20170605/groundtruth'


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
    def __init__(self, src_sequence, num_sequences):
        self.sequence_source = src_sequence
        self.num_sequences = num_sequences

    def load(self):
        n = self.num_sequences if self.num_sequences else '<?>'
        print("Loading %s sequences" % n)

        seq_id2seq = dict()
        for i, seq in enumerate(self.sequence_source):
            sys.stdout.write("\rProcessing sequences\t%s" % i)
            seq_id, seq_seq = self.parse_sequence(seq)
            seq_id2seq[seq_id] = seq_seq

        print("\nFinished loading %s sequences!" % i)

        return seq_id2seq

    def parse_sequence(self, seq):
        return None, None


class FastaFileLoader(SequenceLoader):
    def __init__(self, src_fasta):
        super(FastaFileLoader, self).__init__(src_fasta, None)

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

        n = self.mapping_count if self.mapping_count else '<?>'
        print("Loading %s mappings." % n)

        direct_map, reverse_map = dict(), dict()
        for i, item in enumerate(self.mapping_source):
            sys.stdout.write("\rProcessing mappings\t%s" % i)
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
    def __init__(self, file_src, num_lines):
        super(MappingFileLoader, self).__init__(file_src, num_lines)

    def parse_mapping(self, line):
        return line.strip().split('\t')


class GoAnnotationLoader(MappingLoader):
    def __init__(self, src_annotations, num_annotations, aspect=GoAspect()):
        super(GoAnnotationLoader, self)\
            .__init__(src_annotations, num_annotations)
        self.aspect = aspect

    def parse_mapping(self, entry):
        return None, None


class GoAnnotationFileLoader(GoAnnotationLoader):
    def __init__(self, annotation_file_io, aspect):
        super(GoAnnotationFileLoader, self)\
            .__init__(annotation_file_io, None, aspect)

    def parse_mapping(self, line):
        seq_id, go_id, go_asp = line.strip().split('\t')
        if go_asp == self.aspect:
            return seq_id, go_id
        else:
            return None, None


class GoAnnotationCollectionLoader(GoAnnotationLoader):
    def __init__(self, annotation_cursor, annotation_count, aspect):
        super(GoAnnotationCollectionLoader, self) \
            .__init__(annotation_cursor, annotation_count, aspect)

    def parse_mapping(self, doc):
        seq_id, go_id, go_asp = doc["DB_Object_ID"], doc["GO_ID"], doc["Aspect"]
        if go_asp == self.aspect:
            return seq_id, go_id
        else:
            return None, None


# load_training_data_from_collection(cafa3_cutoff, GoAspect('P'))
def load_training_data_from_collection(cutoff_date, aspect, exp=False):
    query = {"DB": "UniProtKB",
             "Date": {"$lte": cutoff_date}}
    if exp:
        query["Evidence"] = {"$in": exp_codes}
    annot_src = db.goa_uniprot.find(query)
    annot_num = db.goa_uniprot.count(query)
    seq_id2go_id, go_id2seq_id = \
        GoAnnotationCollectionLoader(annot_src, annot_num, aspect).load()
    query = {"_id": {"$in": unique(list(seq_id2go_id.keys())).tolist()}}
    sequence_src = db.uniprot.find(query)
    sequence_num = db.uniprot.count(query)
    seq_id2seq = UniprotCollectionLoader(sequence_src, sequence_num).load()
    return seq_id2seq, seq_id2go_id, go_id2seq_id


def load_training_data_from_files(annots_tsv, seqs_fasta, aspect):
    annot_src = open(annots_tsv, 'r')
    seq_id2go_id, go_id2seq_id = \
        GoAnnotationFileLoader(annot_src, aspect).load()
    fasta_src = parse_fasta(open(seqs_fasta, 'r'), 'fasta')
    seq_id2seq = FastaFileLoader(fasta_src).load()
    return seq_id2seq, seq_id2go_id, go_id2seq_id


def load_cafa3_targets(targets_dir, mapping_dir):
    trg_id2seq, trg_id2seq_id, seq_id2trg_id = dict(), dict(), dict()
    for fasta_file in os.listdir(targets_dir):
        fasta_src = parse_fasta(open(fasta_file, 'r'), 'fasta')
        trg_id2seq.update(FastaFileLoader(fasta_src).load())
    for mapping_file in os.listdir(mapping_dir):
        mapping_src = open(mapping_file, 'r')
        d1, d2 = MappingLoader
        trg_id2seq.update(MappingFileLoader(mapping_src).load())
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


if __name__ == "__main__":

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    sub_dir = cafa3_train_dir = 'CAFA3_training_data'
    if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
        wget_and_unzip(sub_dir, data_dir, cafa3_train_url)
    # sub_dir = cafa3_targets_dir = 'CAFA3_targets'
    # if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
    #     wget_and_unzip(sub_dir, data_dir, cafa3_targets_url)
    # sub_dir = cafa2_targets_dir = 'CAFA-2013-targets'
    # if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
    #     wget_and_unzip(sub_dir, data_dir, cafa2_targets_url)
    # sub_dir = cafa2_data_dir = 'CAFA2Supplementary_data'
    # if not os.path.exists('%s/%s' % (data_dir, sub_dir)):
    #     wget_and_unzip(sub_dir, data_dir, cafa2_data_url)

    cafa3_go_tsv = '%s/%s/uniprot_sprot_exp.txt' % (data_dir, cafa3_train_dir)
    cafa3_train_fasta = '%s/%s/uniprot_sprot_exp.fasta' % (data_dir, cafa3_train_dir)
    load_training_data_from_files(cafa3_go_tsv, cafa3_train_fasta, GoAspect('P'))
    load_training_data_from_collection(cafa3_cutoff, GoAspect('P'))

    cafa3_targets_dir = 'Target Files'
    cafa3_mapping_dir = 'Mapping Files'
    load_cafa3_targets(cafa3_targets_dir, cafa3_mapping_dir)

    cafa2_targets_dir = './CAFA2Supplementary_data/data/CAFA2-targets'
    cafa2_benchmark_dir = './CAFA2Supplementary_data/data/benchmark'

