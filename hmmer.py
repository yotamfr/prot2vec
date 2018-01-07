import os
import sys
import subprocess
from tqdm import tqdm

from Bio import SeqIO, AlignIO
from Bio.Align.AlignInfo import SummaryInfo

from src.python.preprocess import *

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()

import argparse


ASPECT = 'F'
GAP = '-'


def _get_annotated_uniprot(db, limit):
    query = {'DB': 'UniProtKB', 'Evidence': {'$in': exp_codes}}

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    seqid2goid, goid2seqid = GoAnnotationCollectionLoader(s, c, ASPECT).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    return seqid2seq, seqid2goid, goid2seqid


def _run_hmmer_single_thread(sequences, iter=3):
    pbar = tqdm(range(len(sequences)), desc="sequences processed")
    database = "data/Uniprot/uniprot_sprot.fasta"
    outpth = os.path.join(tmp_dir, "align.sto")
    for seqid, seq in sequences.items():
        pbar.update(1)
        cline = "jackhmmer -N %d --acc --noali -A %s - %s" % (iter, outpth, database)

        child = subprocess.Popen(cline,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=(sys.platform != "win32"))

        stdout, _ = child.communicate(input=">%s\n%s" % (seqid, seq))
        assert child.returncode == 0

        info = SummaryInfo(AlignIO.read(outpth, "stockholm"))
        pssm = info.pos_specific_score_matrix(chars_to_ignore=[GAP])

        db.pssm.update_one({
            "_id": seqid}, {
            '$set': {"pssm": pssm.pssm, "seq": seq}
        }, upsert=True)

    pbar.close()


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--limit", type=int, default=None,
                        help="How many sequences for PSSM computation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    lim = args.limit

    seqs, _, _ = _get_annotated_uniprot(db, lim)
    pssm = _run_hmmer_single_thread(seqs)
