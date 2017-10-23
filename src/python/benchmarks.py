
import sys
import datetime

from numpy import unique
from Bio.SeqIO import parse as parse_fasta
from pymongo import MongoClient

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, required=True,
#                     help="Supply working directory for the data")
parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                    help="Supply the URL of MongoDB")
args = parser.parse_args()

client = MongoClient(args.mongo_url)
db_name = 'prot2vec'
db = client[db_name]

cafa2_cutoff = datetime.datetime(2014, 1, 1, 0, 0)
cafa3_cutoff = datetime.datetime(2017, 2, 2, 0, 0)

exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"]


data_dir = '../../data/cafa/CAFA3_training_data'


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
        return "Biological_Process" if aspect == 'P'\
            else "Molecular_Function" if aspect == 'F'\
            else "Cellular_Component" if aspect == 'C'\
            else "unspecified" if not aspect\
            else "unknown"


class Uniprot(object):

    def __init__(self, doc):
        self.uid = doc["_id"]
        self.db_id = doc["db"]
        self.name = doc["entry_name"]
        self.seq = doc["sequence"]

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, val):
        self.__name = val

    @property
    def seq(self):
        return self.__seq

    @seq.setter
    def seq(self, val):
        self.__seq = val

    @property
    def uid(self):
        return self.__uid

    @uid.setter
    def uid(self, val):
        self.__uid = val

    def get_go_terms(self):
        return set(
            map(lambda e: e["GO_ID"], db.goa_uniprot.find({"DB_Object_ID": self.uid}))
        )


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


class GoAnnotationLoader(object):

    def __init__(self, src_annotation, num_annotations):
        self.annotation_source = src_annotation
        self.num_annotation = num_annotations

    def load(self, aspect):

        n = self.num_annotation if self.num_annotation else '<?>'
        print("Loading %s annotations of %s aspect." % (n, aspect))

        seq_id2go_id, go_id2seq_id = dict(), dict()
        for i, item in enumerate(self.annotation_source):
            sys.stdout.write("\rProcessing annotations\t%s" % i)
            seq_id, go_id, go_asp = self.parse_annotation(item)
            if aspect != 'unspecified' and aspect != go_asp:
                continue
            try:
                if go_id in go_id2seq_id:
                    go_id2seq_id[go_id].add(seq_id)
                else:
                    go_id2seq_id[go_id] = {seq_id}
                if seq_id in seq_id2go_id:
                    seq_id2go_id[seq_id].add(go_id)
                else:
                    seq_id2go_id[seq_id] = {go_id}
            except TypeError:
                pass

        m = sum(map(len, seq_id2go_id.values()))
        print("\nFinished loading %s annotations!" % m)

        return seq_id2go_id, go_id2seq_id

    def parse_annotation(self, annot):
        return None, None, None


class GoAnnotationFileLoader(GoAnnotationLoader):

    def __init__(self, annotation_tsv):
        super(GoAnnotationFileLoader, self).__init__(annotation_tsv, None)

    def parse_annotation(self, line):
        return line.strip().split('\t')


class GoAnnotationCollectionLoader(GoAnnotationLoader):

    def __init__(self, annotation_cursor, annotation_count):
        super(GoAnnotationCollectionLoader, self)\
            .__init__(annotation_cursor, annotation_count)

    def parse_annotation(self, doc):
        return doc["DB_Object_ID"], doc["GO_ID"], doc["Aspect"]


# load_training_data_from_collection(cafa3_cutoff, GoAspect('P'))
def load_training_data_from_collection(cutoff_date, aspect, exp=False):
    query = {"DB": "UniProtKB",
           "Date": {"$lte": cutoff_date}}
    if exp:
        query["Evidence"] = {"$in": exp_codes}
    annot_src = db.goa_uniprot.find(query)
    annot_num = db.goa_uniprot.count(query)
    seq_id2go_id, go_id2seq_id = \
        GoAnnotationCollectionLoader(annot_src, annot_num).load(aspect)
    query = {"_id": {"$in": unique(list(seq_id2go_id.keys())).tolist()}}
    sequence_src = db.uniprot.find(query)
    sequence_num = db.uniprot.count(query)
    seq_id2seq = UniprotCollectionLoader(sequence_src, sequence_num).load()
    return seq_id2seq, seq_id2go_id, go_id2seq_id


# tsv = '%s/uniprot_sprot_exp.txt' % data_dir
# fasta = '%s/uniprot_sprot_exp.fasta' % data_dir
# load_training_data_from_files(tsv, fasta, GoAspect('P'))
def load_training_data_from_files(annots_tsv, seqs_fasta, aspect):
    annot_src = open(annots_tsv, 'r')
    seq_id2go_id, go_id2seq_id = \
        GoAnnotationFileLoader(annot_src).load(aspect)
    fasta_src = parse_fasta(open(seqs_fasta, 'r'), 'fasta')
    seq_id2seq = FastaFileLoader(fasta_src).load()
    return seq_id2seq, seq_id2go_id, go_id2seq_id


def load_target_files(datadir):
    pass


def load_benchmark_files(datadir):
    pass


if __name__ == "__main__":
    tsv = '%s/uniprot_sprot_exp.txt' % data_dir
    fasta = '%s/uniprot_sprot_exp.fasta' % data_dir
    load_training_data_from_files(tsv, fasta, GoAspect('P'))

    load_training_data_from_collection(cafa3_cutoff, GoAspect('P'))
