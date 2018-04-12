import os
import sys
import subprocess

import threading
from tqdm import tqdm

from Bio.Seq import Seq as BioSeq
from Bio import SearchIO
from Bio.SeqRecord import SeqRecord

from Bio.Blast.Applications import NcbiblastpCommandline

from src.python.preprocess2 import *

from itertools import cycle
import matplotlib.pyplot as plt

from tempfile import gettempdir
tmp_dir = gettempdir()
out_dir = "./Data"

from concurrent.futures import ThreadPoolExecutor

import argparse

import pickle

import datetime

NUM_CPU = 32

E = ThreadPoolExecutor(NUM_CPU)

EVAL = 10e4


def _prepare_blast(sequences):
    timestamp = datetime.date.today().strftime("%m-%d-%Y")
    blastdb_pth = os.path.join(tmp_dir, 'blast-%s' % timestamp)
    records = [SeqRecord(BioSeq(seq), id) for id, seq in sequences.items()]
    SeqIO.write(records, open(blastdb_pth, 'w+'), "fasta")
    os.system("makeblastdb -in %s -dbtype prot" % blastdb_pth)
    return blastdb_pth


# def _blast(target_fasta, database_pth, topn=None, cleanup=True):
#     seqid = target_fasta.id
#     query_pth = os.path.join(tmp_dir, "%s.fas" % seqid)
#     output_pth = os.path.join(tmp_dir, "%s.out" % seqid)
#     SeqIO.write(target_fasta, open(query_pth, 'w+'), "fasta")
#     cline = NcbiblastpCommandline(query=query_pth, db=database_pth, out=output_pth,
#                                   outfmt=5, evalue=EVAL, remote=False, ungapped=False)
#     child = subprocess.Popen(str(cline), stderr=subprocess.PIPE,
#                              universal_newlines=True, shell=(sys.platform != "win32"))
#     handle, _ = child.communicate()
#     assert child.returncode == 0
#     blast_qresult = SearchIO.read(output_pth, 'blast-xml')
#     distances = {}
#     for hsp in blast_qresult.hsps[:topn]:
#         ident = hsp.ident_num / hsp.hit_span
#         if hsp.hit.id == seqid:
#             assert ident == 1.0
#         distances[hsp.hit.id] = ident
#     if cleanup:
#         os.remove(query_pth)
#         os.remove(output_pth)
#     return distances


def _blast(target_fasta, database_pth, evalue=EVAL):
    seqid = target_fasta.id
    query_pth = os.path.join(tmp_dir, "%s.fas" % seqid)
    output_pth = os.path.join(tmp_dir, "%s.tsv" % seqid)
    SeqIO.write(target_fasta, open(query_pth, 'w+'), "fasta")
    cline = "blastp -db %s -query %s -outfmt 6 -out %s -evalue %d 1>/dev/null 2>&1" \
            % (database_pth, query_pth, output_pth, evalue)
    assert os.WEXITSTATUS(os.system(cline)) == 0
    with open(output_pth, 'r') as f:
        hits = [HSP(line.split('\t')) for line in f.readlines()]
    os.remove(query_pth)
    os.remove(output_pth)
    return hits


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def compute_blast_parallel(uid2seq, db_pth, collection):
    def parallel_blast(db_pth):
        return lambda seq_record: (seq_record, _blast(seq_record, db_pth))
    pbar = tqdm(range(len(uid2seq)), desc="sequences processed")
    inputs = [SeqRecord(BioSeq(seq), uid) for uid, seq in uid2seq.items()]
    for i, (seq, hits) in enumerate(E.map(parallel_blast(db_pth), inputs)):
        for hsp in hits:
            collection.update_one({"_id": hsp.uid}, {"$set": vars(hsp)}, upsert=True)
        pbar.update(1)
    pbar.close()


def to_fasta(seq_map, out_file):
    sequences = []
    for unipid, seq in seq_map.items():
        sequences.append(SeqRecord(BioSeq(seq), unipid))
    SeqIO.write(sequences, open(out_file, 'w+'), "fasta")


def load_nature_repr_set(db):
    def to_fasta(seq_map, out_file):
        sequences = []
        for unipid, seq in seq_map.items():
            sequences.append(SeqRecord(BioSeq(seq), unipid))
        SeqIO.write(sequences, open(out_file, 'w+'), "fasta")
    repr_pth, all_pth = 'Data/sp.nr.70', 'Data/sp.fasta'
    fasta_fname = 'Data/sp.nr.70'
    if not os.path.exists(repr_pth):
        query = {"db": "sp"}
        num_seq = db.uniprot.count(query)
        src_seq = db.uniprot.find(query)
        sp_seqs = UniprotCollectionLoader(src_seq, num_seq).load()
        to_fasta(sp_seqs, all_pth)
        os.system("cdhit/cd-hit -i %s -o %s -c 0.7 -n 5" % (all_pth, repr_pth))
    num_seq = count_lines(fasta_fname, sep=bytes('>', 'utf8'))
    fasta_src = parse_fasta(open(fasta_fname, 'r'), 'fasta')
    seq_map = FastaFileLoader(fasta_src, num_seq).load()
    all_seqs = [Seq(uid, str(seq)) for uid, seq in seq_map.items()]
    return all_seqs


# def get_blast_metric(blast_mat_pth, nature_sequence, sample_size=10e2):
#
#     with open(blast_mat_pth, 'rb') as f:
#         dist_mat = pickle.load(f)
#
#     seq_map = {seq.uid: seq.seq for seq in nature_sequence}
#
#     all_sequences = list(seq_map.keys())
#     prior_sequence_identity = 0.0
#     for i in range(int(sample_size)):
#         uid1 = np.random.choice(all_sequences)
#         uid2 = np.random.choice(all_sequences)
#         while uid1 == uid2:
#             uid2 = np.random.choice(all_sequences)
#         seq1 = Seq(uid1, seq_map[uid1], aa20=True)
#         seq2 = Seq(uid2, seq_map[uid2], aa20=True)
#         assert seq1 != seq2
#         ident = sequence_identity(seq1, seq2)
#         assert 0.0 <= ident <= 1.0
#         prior_sequence_identity += ident
#         sys.stdout.write("\r{0:.0f}%".format(100.0 * i / sample_size))
#     prior_sequence_identity /= sample_size
#
#     print(prior_sequence_identity)
#
#     def blast_seq_identity(sequence1, sequence2):
#         assert sequence1 != sequence2
#         try:
#             return dist_mat[sequence1.uid][sequence2.uid]
#         except KeyError:
#             try:
#                 return dist_mat[sequence1.uid][sequence2.uid]
#             except KeyError:
#                 return prior_sequence_identity
#
#     return blast_seq_identity


class ThreadSafeDict(dict) :
    def __init__(self, * p_arg, ** n_arg) :
        dict.__init__(self, * p_arg, ** n_arg)
        self._lock = threading.Lock()

    def __enter__(self) :
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback) :
        self._lock.release()


class HSP(object):
    def __init__(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            qseqid, sseqid, pident, length, mismatch, gapopen, qstart, qend, sstart, send, evalue, bitscore = data
            self.qseqid = qseqid
            self.sseqid = sseqid
            self.pident = float(pident)
            self.length = float(length)
            self.mismatch = float(mismatch)
            self.gapopen = float(gapopen)
            self.qstart = float(qstart)
            self.qend = float(qend)
            self.sstart = float(sstart)
            self.send = float(send)
            self.evalue = float(evalue)
            self.bitscore = float(bitscore)
        elif isinstance(data, dict):
            for k, v in data.items():
                setattr(self, k, v)
        else:
            raise ValueError("HSP.__init__ cannot handle %s" % type(data))

    @property
    def uid(self):
        hsp = self
        templ = "%s_%s_%d_%d_%d_%d_%d"
        args = (hsp.qseqid, hsp.sseqid, hsp.length, hsp.qstart, hsp.qend, hsp.sstart, hsp.send)
        return templ % args


class Seq(object):

    def __init__(self, uid, seq, aa20=True):

        if aa20:
            self.seq = seq.replace('U', 'C').replace('O', 'K')\
                .replace('X', np.random.choice(amino_acids))\
                .replace('B', np.random.choice(['N', 'D']))\
                .replace('Z', np.random.choice(['E', 'Q']))
        else:
            self.seq = seq

        self.uid = uid
        self.msa = None
        self.f = dict()

    def __hash__(self):
        return hash(self.uid)

    def __repr__(self):
        return "Seq(%s, %s)" % (self.uid, self.seq)

    def __eq__(self, other):
        if isinstance(other, Seq):
            return self.uid == other.uid
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.seq)


def load_object(pth):
    with open(pth, 'rb') as f:
        loaded_dist_mat = pickle.load(f)
        assert len(loaded_dist_mat) > 0
    return loaded_dist_mat


class BLAST(object):

    def __init__(self, collection):
        self.collection = collection
        self.blast_cache = {}

    def load_precomputed(self, targets):
        collection = self.collection
        blast_cache = self.blast_cache
        pbar = tqdm(range(len(targets)), desc="blast hits loaded")
        for tgt in targets:
            hits = map(HSP, collection.find({"qseqid": tgt.uid}))
            blast_cache[tgt] = list(hits)
            pbar.update(1)
        pbar.close()

    def get_hits(self, seq1, seq2):
        collection = self.collection
        hits = list(map(HSP, collection.find({"qseqid": seq1.uid, "sseqid": seq2.uid})))
        return hits

    def __getitem__(self, uid):
        return self.blast_cache[uid]


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--db_name", type=str, default='prot2vec', choices=['prot2vec', 'prot2vec2'],
                        help="The name of the DB to which to write the data.")
    parser.add_argument("--aspect", type=str, default='F', choices=['F', 'P', 'C'],
                        help="The name of the DB to which to write the data.")
    parser.add_argument('--load', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    from pymongo import MongoClient
    client = MongoClient(args.mongo_url)
    db = client[args.db_name]
    asp = args.aspect   # molecular function

    onto = get_ontology(asp)
    t0 = datetime.datetime(2014, 1, 1, 0, 0)
    t1 = datetime.datetime(2014, 9, 1, 0, 0)

    print("Indexing Data...")
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, asp)
    print("Loading Train Data...")
    uid2seq_trn, _, _ = trn_stream.to_dictionaries(propagate=True)
    print("Loading Validation Data...")
    uid2seq_tst, _, _ = tst_stream.to_dictionaries(propagate=True)

    db_pth = _prepare_blast(uid2seq_trn)

    if args.load:
        # timestamp = datetime.date.today().strftime("%m-%d-%Y")
        # pth = "Data/blast_%s_hits_%s" % (asp, timestamp)
        pth = args.load  # Data/blast_F_hits_04-11-2018
        assert os.path.exists(pth)
        print("Loading %s" % pth)
        blast_hsp_matrix = load_object(pth)
        print("Loaded %d records" % (len(blast_hsp_matrix)))
        pbar = tqdm(range(len(blast_hsp_matrix)), desc="hits processed")
        for i, (_, hsp) in enumerate(blast_hsp_matrix.items()):
            db.blast.update_one({"_id": hsp.uid}, {"$set": vars(hsp)}, upsert=True)
            pbar.update(1)
        pbar.close()

    else:
        blast_hsp_matrix = db.blast
        db.blast.create_index("qseqid")
        db.blast.create_index("sseqid")

        compute_blast_parallel(uid2seq_tst, db_pth, blast_hsp_matrix)
        # save_object(blast_hsp_matrix, pth)
        compute_blast_parallel(uid2seq_trn, db_pth, blast_hsp_matrix)
        # save_object(blast_hsp_matrix, pth)

        nature_set = load_nature_repr_set(db)
        uid2seq_nature = {seq.uid: seq.seq for seq in nature_set}
        compute_blast_parallel(uid2seq_nature, db_pth, blast_hsp_matrix)
        # save_object(blast_hsp_matrix, pth)
