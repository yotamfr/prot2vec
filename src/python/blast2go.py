import os

import subprocess

from tqdm import tqdm

from Bio.Seq import Seq
from Bio import SeqIO, SearchIO
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastpCommandline

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


def get_targets(cutoff, asp, limit=None):

    q = {'DB': 'UniProtKB',
         'Evidence': {'$in': exp_codes},
         'Date':  {"$gt": cutoff},
         'Aspect': asp}

    c = limit if limit else db.goa_uniprot.count(q)
    s = db.goa_uniprot.find(q)
    if limit: s = s.limit(limit)

    seqid2goid, goid2seqid = GoAnnotationCollectionLoader(s, c, asp).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()
    onto = get_ontology(asp)

    for k, v in seqid2goid.items():
        seqid2goid[k] = onto.augment(v)

    return seqid2seq, seqid2goid, goid2seqid


def blast2go_init(cutoff, asp, limit=None):

    stream = db.goa_uniprot.find({
        'DB': 'UniProtKB',
        'Evidence': {'$in': exp_codes},
        'Date':  {"$lte": cutoff},
        'Aspect': asp
    })
    if limit: stream = stream.limit(limit)

    uniprot_ids = list(set(map(lambda doc: doc['DB_Object_ID'], stream)))
    stream = db.uniprot.find({'_id': {'$in': uniprot_ids}})
    records = [SeqRecord(Seq(doc['sequence']), doc['_id']) for doc in stream]
    blastdb_pth = os.path.join(tmp_dir, 'blast2go-%s' % GoAspect(asp))
    SeqIO.write(records, open(blastdb_pth, 'w+'), "fasta")

    os.system("makeblastdb -in %s -dbtype prot" % blastdb_pth)
    return uniprot_ids


def blast2go_predict(targets, asp, limit=None):

    query_pth = os.path.join(tmp_dir, 'query.fasta')
    output_pth = os.path.join(tmp_dir, "blastp.out")
    database_pth = os.path.join(tmp_dir, 'blast2go-%s' % GoAspect(asp))

    records = [SeqRecord(Seq(seq), id) for id, seq in targets.items()]

    if limit: records = records[:limit]
    pbar = tqdm(range(len(records)), desc="targets processed")

    for record in records:

        SeqIO.write(record, open(query_pth, 'w+'), "fasta")

        cline = NcbiblastpCommandline(query=query_pth, db=database_pth, out=output_pth,
                                      outfmt=5, evalue=0.001, remote=False, ungapped=False)

        child = subprocess.Popen(str(cline),
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=(sys.platform != "win32"))

        handle, _ = child.communicate()
        assert child.returncode == 0

        blast_qresult = SearchIO.read(output_pth, 'blast-xml')

        # print("%s %s" % (blast_qresult.id, blast_qresult.description))
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    asp = 'F'

    blast2go_ids = set(blast2go_init(cafa3_cutoff, asp))

    targets, truth, _ = get_targets(cafa3_cutoff, asp)
    targets = {k: v for k, v in targets.items() if k not in blast2go_ids}

    blast2go_predict(targets, asp, 100)



