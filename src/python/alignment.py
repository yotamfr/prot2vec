#!/usr/bin/env python

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
import Levenshtein as lv

from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client["prot2vec"]


def sequence_identity_by_id(id1, id2, collection):
    dom1 = collection.find_one({"_id": id1})
    dom2 = collection.find_one({"_id": id2})
    return sequence_identity(
        dom1["sequence"],
        dom2["sequence"]
    )


def align(seq1, seq2, matrix=matlist.blosum62):
    # return pairwise2.align.globaldx(seq1, seq2, matrix)
    return pairwise2.align.globalxx(seq1, seq2)


def sequence_identity(sequence1, sequence2):
    alignment = align(sequence1, sequence2)
    return lv.ratio(alignment[0][0], alignment[0][1])


def main():
    print(sequence_identity_by_id('200LA', '114LA', db.pdb))

if __name__ == "__main__":
    main()
