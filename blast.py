import os
import sys
import subprocess

import threading
from tqdm import tqdm

from Bio.Seq import Seq as BioSeq
from Bio import SearchIO
from Bio.SeqRecord import SeqRecord

from src.python.preprocess2 import *

from tempfile import gettempdir
tmp_dir = gettempdir()
out_dir = "./Data"

from concurrent.futures import ThreadPoolExecutor

import argparse

import pickle

import datetime

NUM_CPU = 8

E = ThreadPoolExecutor(NUM_CPU)


def prepare_blast(sequences):
    timestamp = datetime.date.today().strftime("%m-%d-%Y")
    blastdb_pth = os.path.join(tmp_dir, 'blast-%s' % timestamp)
    records = [SeqRecord(BioSeq(seq), uid) for uid, seq in sequences.items()]
    SeqIO.write(records, open(blastdb_pth, 'w+'), "fasta")
    os.system("makeblastdb -in %s -dbtype prot" % blastdb_pth)
    return blastdb_pth


def _blast(target_fasta, database_pth, evalue):
    seqid = target_fasta.id
    query_pth = os.path.join(tmp_dir, "%s.fas" % seqid)
    output_pth = os.path.join(tmp_dir, "%s.tsv" % seqid)
    SeqIO.write(target_fasta, open(query_pth, 'w+'), "fasta")
    cline = "blastp -db %s -query %s -outfmt 6 -out %s -evalue %s 1>/dev/null 2>&1" \
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


def parallel_blast(db_pth, evalue):
    return lambda seq_record: (seq_record, _blast(seq_record, db_pth, evalue))


def compute_blast_parallel(uid2seq, db_pth, collection, evalue):
    pbar = tqdm(range(len(uid2seq)), desc="sequences processed")
    inputs = [SeqRecord(BioSeq(seq), uid) for uid, seq in uid2seq.items()]
    for i, (seq, hits) in enumerate(E.map(parallel_blast(db_pth, evalue), inputs)):
        for hsp in hits:
            collection.update_one({"_id": hsp.uid}, {"$set": vars(hsp)}, upsert=True)
        pbar.update(1)
    pbar.close()


def predict_blast_parallel(queries, seqid2go, db_pth, evalue):
    pbar = tqdm(range(len(queries)), desc="queries processed")
    inputs = [SeqRecord(BioSeq(tgt.seq), tgt.uid) for tgt in queries]
    query2hits = {}
    for i, (query, hits) in enumerate(E.map(parallel_blast(db_pth, evalue), inputs)):
        query2hits[query.id] = hits
        pbar.update(1)
    pbar.close()
    query2preds = {}
    pbar = tqdm(range(len(query2hits)), desc="queries processed")
    for i, (qid, hits) in enumerate(query2hits.items()):
        pbar.update(1)
        query2preds[qid] = {}
        if len(hits) == 0:
            continue
        for hsp in hits:
            for go in seqid2go[hsp.sseqid]:
                if go in query2preds[qid]:
                    query2preds[qid][go].append(hsp)
                else:
                    query2preds[qid][go] = [hsp]
    pbar.close()
    return query2preds


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
    repr_pth, all_pth = '%s/sp.nr.70' % out_dir, '%s/sp.fasta' % out_dir
    fasta_fname = '%s/sp.nr.70' % out_dir
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
        self.qid2hsp = {}

    def sort_by_count(self, targets, subject=False):
        collection = self.collection
        target2count = {}
        key = "sseqid" if subject else "qseqid"
        pbar = tqdm(range(len(targets)), desc="targets processed")
        for tgt in targets:
            target2count[tgt] = collection.count({key: tgt.uid})
            pbar.update(1)
        pbar.close()
        seqs = sorted(targets, key=lambda t: -target2count[t])      # reverse order
        return seqs

    def load_precomputed(self, targets):
        collection = self.collection
        pbar = tqdm(range(len(targets)), desc="targets loaded")
        for tgt in targets:
            self[tgt] = {}
            for hsp in map(HSP, collection.find({"qseqid": tgt.uid})):
                subj = hsp.sseqid
                try:
                    old = self[tgt][subj]
                    if old.evalue > hsp.evalue:
                        self[tgt][subj] = hsp
                except KeyError:
                    self[tgt][subj] = hsp
            pbar.update(1)
        pbar.close()

    def get_hits(self, seq1, seq2):
        collection = self.collection
        hits = list(map(HSP, collection.find({"qseqid": seq1.uid, "sseqid": seq2.uid})))
        return hits

    def blastp(self, seq1, seq2, evalue=10e6):
        query_pth = os.path.join(tmp_dir, "%s.seq" % seq1.uid)
        subject_pth = os.path.join(tmp_dir, "%s.seq" % seq2.uid)
        output_pth = os.path.join(tmp_dir, "%s_%s.out" % (seq1.uid, seq2.uid))
        SeqIO.write(SeqRecord(BioSeq(seq1.seq), seq1.uid), open(query_pth, 'w+'), "fasta")
        SeqIO.write(SeqRecord(BioSeq(seq2.seq), seq2.uid), open(subject_pth, 'w+'), "fasta")
        cline = "blastp -query %s -subject %s -outfmt 6 -out %s -evalue %s 1>/dev/null 2>&1" \
                % (query_pth, subject_pth, output_pth, evalue)
        if os.WEXITSTATUS(os.system(cline)) != 0:
            print("BLAST failed unexpectedly (%s, %s)" % (seq1.uid, seq2.uid))
            return HSP([seq1.uid, seq2.uid, 0., 0., 0., 0., 0., 0., 0., 0., evalue * 10, 10e-6])
        assert os.path.exists(output_pth)
        with open(output_pth, 'r') as f:
            hits = [HSP(line.split('\t')) for line in f.readlines()]
            if len(hits) == 0:
                return HSP([seq1.uid, seq2.uid, 0., 0., 0., 0., 0., 0., 0., 0., evalue * 10, 10e-6])
            hsp = hits[np.argmin([h.evalue for h in hits])]
            self.collection.update_one({"_id": hsp.uid}, {"$set": vars(hsp)}, upsert=True)
        return hsp

    def __getitem__(self, seq):
        return self.qid2hsp[seq.uid]

    def __setitem__(self, seq, val):
        self.qid2hsp[seq.uid] = val

    def __contains__(self, seq):
        return seq in self.qid2hsp[seq.uid]


def cleanup():
    files = os.listdir(tmp_dir)
    for file in files:
        if file.endswith(".fas") or file.endswith(".tsv"):
            os.remove(os.path.join(tmp_dir, file))


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--db_name", type=str, default='prot2vec', choices=['prot2vec', 'prot2vec2'],
                        help="The name of the DB to which to write the data.")
    parser.add_argument("--aspect", type=str, default='F', choices=['F', 'P', 'C'],
                        help="The name of the DB to which to write the data.")
    parser.add_argument("--mode", type=str, default='predict', choices=['comp', 'load', 'predict'],
                        help="In which mode to do you want me to work?")
    parser.add_argument("--evalue", type=int, default=10e3,
                        help="Set evalue threshold for BLAST")


if __name__ == "__main__":

    cleanup()

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
    # t0 = datetime.datetime(2017, 1, 1, 0, 0)
    # t1 = datetime.datetime.utcnow()

    print("Indexing Data...")
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, asp)
    print("Loading Train Data...")
    uid2seq_trn, uid2go_trn, _ = trn_stream.to_dictionaries(propagate=True)
    print("Loading Validation Data...")
    uid2seq_tst, uid2go_tst, _ = tst_stream.to_dictionaries(propagate=True)

    db_pth = prepare_blast(uid2seq_trn)
    timestamp = datetime.date.today().strftime("%m-%d-%Y")

    if args.mode == "load":
        pth = "%s/blast_%s_hits_%s" % (out_dir, asp, timestamp)
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
        cleanup()

    elif args.mode == "comp":
        blast_hsp_matrix = db.blast
        db.blast.create_index("qseqid")
        db.blast.create_index("sseqid")

        compute_blast_parallel(uid2seq_tst, db_pth, blast_hsp_matrix, args.evalue)
        compute_blast_parallel(uid2seq_trn, db_pth, blast_hsp_matrix, args.evalue)

        nature_set = load_nature_repr_set(db)
        uid2seq_nature = {seq.uid: seq.seq for seq in nature_set}
        compute_blast_parallel(uid2seq_nature, db_pth, blast_hsp_matrix, args.evalue)
        cleanup()

    elif args.mode == "predict":
        # evalue = 0.001
        queries = [Seq(uid, seq) for uid, seq in uid2seq_tst.items()]
        predictions = predict_blast_parallel(queries, uid2go_trn, db_pth, args.evalue)
        save_object(predictions, "%s/blast_%s_hsp" % (out_dir, GoAspect(asp)))
        save_object(uid2go_tst, "%s/gt_%s" % (out_dir, GoAspect(asp)))
        cleanup()

    else:
        print("unknown mode")
        exit(0)
