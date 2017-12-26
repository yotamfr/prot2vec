
import os

from tqdm import tqdm


from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import argparse

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()

from src.python.preprocess import *


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")


def load_datasets():
    mf, _ = load_data(db, asp='F', codes=exp_codes)
    cc, _ = load_data(db, asp='C', codes=exp_codes)
    bp, _ = load_data(db, asp='P', codes=exp_codes)
    return mf, cc, bp


def blast2go(asp = 'F', lim = 100):
    stream = db.goa_uniprot.find({'Evidence': {'$in': exp_codes}, 'Aspect': asp})
    if lim: stream = stream.limit(lim)

    uniprot_ids = list(set(map(lambda doc: doc['DB_Object_ID'], stream)))

    query = {'_id': {'$in': uniprot_ids}}
    stream = db.uniprot.find(query)
    count = db.uniprot.count(query)

    records = [SeqRecord(Seq(doc['sequence']), doc['_id']) for doc in stream]
    blastdb_fname = '%s/%s_seq.fasta' % (tmp_dir, GoAspect(asp))
    f = open(blastdb_fname, 'w+')
    SeqIO.write(records, f, "fasta")

    os.system("makeblastdb -in %s -dbtype prot" % blastdb_fname)

    pbar = tqdm(range(count), desc="sequences loaded")
    for doc in stream:
        pbar.update(1)
    pbar.close()

    print(uniprot_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    blast2go()



