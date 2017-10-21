import sys
import datetime
from Bio import SeqIO
from pymongo import MongoClient
from tqdm import tqdm

import argparse

client = MongoClient('mongodb://localhost:27017/')
dbname = 'prot2vec'
db = client[dbname]


def add_single_uniprot(fasta):

    header, sequence = fasta.id, str(fasta.seq)
    dbname, unique_identifier, entry_name = \
        header.split(' ')[0].split('|')

    prot = {
        "primary_accession": unique_identifier,
        "db": dbname,
        "entry_name": entry_name,
        "sequence": sequence,
        "length": len(sequence),
        "created_at": datetime.datetime.utcnow(),
        "header": header
    }

    db.uniprot.update_one({
        "_id": unique_identifier}, {
        "$set": prot
    }, upsert=True)


def load_fasta(src_fasta, start=db.uniprot.count({})):

    print("Loading %s" % src_fasta)

    num_seq = 0
    fasta_sequences = SeqIO.parse(open(src_fasta), 'fasta')
    for _ in fasta_sequences:
        sys.stdout.write("\rCounting sequences\t%s" % num_seq)
        num_seq += 1

    db.uniprot.create_index("db")
    db.uniprot.create_index("entry_name")
    db.uniprot.create_index("primary_accession")

    print("\nLoading %s Uniprot sequences to %s ...\n" % (num_seq, dbname))

    fasta_sequences = SeqIO.parse(open(src_fasta), 'fasta')
    for i in tqdm(range(num_seq), desc="sequences already processed"):
        if i < start:
            continue
        add_single_uniprot(next(fasta_sequences))

    print("\nFinished!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta", type=str, help="Give full path to uniprot fasta file")
    args = parser.parse_args()
    load_fasta(args.fasta, 0)

if __name__ == "__main__":
    main()
