#!/usr/bin/python3

import sys
import datetime
from Bio.SeqIO import parse
from pymongo import MongoClient

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", type=str, help="Give full path to uniprot fasta file")
parser.add_argument("--collection", type=str, choices=['uniprot', 'sprot'],
                    default="uniprot", help="Give collection name.")

args = parser.parse_args()

client = MongoClient('mongodb://localhost:27017/')
db_name = 'prot2vec'
collection = client[db_name][args.collection]


def add_single_uniprot(fasta):

    header, sequence = fasta.id, str(fasta.seq)
    db, unique_identifier, entry_name = \
        header.split(' ')[0].split('|')

    prot = {
        "primary_accession": unique_identifier,
        "db": db,
        "entry_name": entry_name,
        "sequence": sequence,
        "length": len(sequence),
        "created_at": datetime.datetime.utcnow(),
        "header": header
    }

    collection.update_one({
        "_id": unique_identifier}, {
        "$set": prot
    }, upsert=True)


def load_fasta(src_fasta, start=collection.count({})):

    print("Loading %s" % src_fasta)

    collection.create_index("db")
    collection.create_index("entry_name")
    collection.create_index("primary_accession")

    with open(src_fasta, 'r') as f:

        fasta_sequences = parse(f, 'fasta')

        for i, seq in enumerate(fasta_sequences):

            if i % 100 == 0:
                sys.stdout.write("\rProcessed sequences\t%s" % i)

            if i < start: continue

            add_single_uniprot(seq)

    print("\nFinished!")


def main():
    load_fasta(args.fasta, 0)


if __name__ == "__main__":
    main()
