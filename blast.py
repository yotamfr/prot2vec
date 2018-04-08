import os
import sys
import subprocess

from tqdm import tqdm

from Bio.Seq import Seq
from Bio import SearchIO
from Bio.SeqRecord import SeqRecord

from Bio.Blast.Applications import NcbiblastpCommandline

from src.python.preprocess2 import *

from itertools import cycle
import matplotlib.pyplot as plt

from tempfile import gettempdir
tmp_dir = gettempdir()

from concurrent.futures import ThreadPoolExecutor

import argparse

import pickle

import datetime

NUM_CPU = 8

E = ThreadPoolExecutor(NUM_CPU)


def _prepare_blast(sequences):
    timestamp = datetime.date.today().strftime("%m-%d-%Y_%I:%m%p")
    blastdb_pth = os.path.join(tmp_dir, 'blast-%s' % timestamp)
    records = [SeqRecord(Seq(seq), id) for id, seq in sequences.items()]
    SeqIO.write(records, open(blastdb_pth, 'w+'), "fasta")
    os.system("makeblastdb -in %s -dbtype prot" % blastdb_pth)
    return blastdb_pth


def _blast(target_fasta, database_pth, topn=None, cleanup=True):
    seqid = target_fasta.id
    query_pth = os.path.join(tmp_dir, "%s.fas" % seqid)
    output_pth = os.path.join(tmp_dir, "%s.out" % seqid)
    SeqIO.write(target_fasta, open(query_pth, 'w+'), "fasta")
    cline = NcbiblastpCommandline(query=query_pth, db=database_pth, out=output_pth,
                                  outfmt=5, evalue=0.001, remote=False, ungapped=False)
    child = subprocess.Popen(str(cline), stderr=subprocess.PIPE,
                             universal_newlines=True, shell=(sys.platform != "win32"))
    handle, _ = child.communicate()
    assert child.returncode == 0
    blast_qresult = SearchIO.read(output_pth, 'blast-xml')
    distances = {}
    for hsp in blast_qresult.hsps[:topn]:
        ident = hsp.ident_num / hsp.hit_span
        if hsp.hit.id == seqid:
            assert ident == 1.0
        distances[hsp.hit.id] = ident
    if cleanup:
        os.remove(query_pth)
        os.remove(output_pth)
    return distances


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def compute_blast_parallel(uid2seq, dist_mat):
    def parallel_blast(db_pth):
        return lambda seq_record: (seq_record, _blast(seq_record, db_pth))
    pbar = tqdm(range(len(uid2seq)), desc="sequences processed")
    inputs = [SeqRecord(Seq(seq), uid) for uid, seq in uid2seq.items()]
    for i, (seq, distances) in enumerate(E.map(parallel_blast(db_pth), inputs)):
        dist_mat[seq.id] = distances
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prot2vec']

    asp = 'F'   # molecular function
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
    dist_mat = {}

    compute_blast_parallel(uid2seq_trn, dist_mat)
    compute_blast_parallel(uid2seq_tst, dist_mat)

    timestamp = datetime.date.today().strftime("%m-%d-%Y_%I:%m%p")
    save_object(dist_mat, "Data/blast_dist_matrix_%s" % timestamp)

    with open("Data/blast_dist_matrix_%s" % timestamp, 'rb') as f:
        loaded_dist_mat = pickle.load(f)
        assert len(loaded_dist_mat) == len(dist_mat)
