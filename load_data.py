import os
import datetime
from Bio.UniProt.GOA import gafiterator
from Bio.SeqIO import parse as parse_fasta
from pymongo import MongoClient
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=str, default="./data/raw",
                    help="Supply working directory for the data")
parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                    help="Supply the URL of MongoDB")
parser.add_argument("c", "--collection", type=str, required=True,
                    choices=['uniprot', 'sprot', 'goa_uniprot', 'goa_pdb'],
                    help="The name of the collection that you want load.")
parser.add_argument('--exp', action='store_true', default=False,
                    help="Load only experimentally validated annotations.")
args = parser.parse_args()

client = MongoClient(args.mongo_url)
db_name = 'prot2vec'
db = client[db_name]

exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"]


def blocks(files, size=8192*1024):
    while True:
        buffer = files.read(size)
        if not buffer:
            break
        yield buffer


def count_lines(fpath, sep='\n'):
    with open(fpath, "rb") as f:
        return sum(bl.count(sep) for bl in blocks(f))


def wget(url, fname, datadir):
    os.system('wget -O %s/%s.gz %s/%s.gz' % (datadir, fname, url, fname))


def unzip(datadir, fname):
    fpath = '%s/%s' % (datadir, fname)
    os.system('gunzip -c %s.gz > %s' % (fpath, fpath))


# load Gene Ontology Annotations in a flat structure
def load_gaf(filename, collection, start=0):

    print("Loading: %s" % filename)

    collection.create_index("DB_Object_ID")
    collection.create_index("DB")
    collection.create_index("GO_ID")
    collection.create_index("Evidence")
    collection.create_index("Aspect")
    collection.create_index("Date")

    n = count_lines(filename, sep=bytes('\n', 'ascii'))
    pbar = tqdm(range(n), desc="annotations loaded")

    with open(filename, 'r') as handler:

        goa_iterator = gafiterator(handler)

        for i, data in enumerate(goa_iterator):

            if i < start or (args.exp and data['Evidence'] not in exp_codes):
                pbar.update(1)
                continue

            date = datetime.datetime.strptime(data['Date'], "%Y%m%d").date()

            json = {
                "DB_Object_ID": data['DB_Object_ID'],
                "DB_Object_Symbol": data['DB_Object_Symbol'],
                "With": data['With'],
                "Assigned_By": data['Assigned_By'],
                "Annotation_Extension": data['Annotation_Extension'],
                "Gene_Product_Form_ID": data['Gene_Product_Form_ID'],
                "DB:Reference": data['DB:Reference'],
                "GO_ID": data['GO_ID'],
                "Qualifier": data['Qualifier'],
                "Date": datetime.datetime.fromordinal(date.toordinal()),
                "DB": data['DB'],
                "created_at": datetime.datetime.utcnow(),
                "DB_Object_Name": data['DB_Object_Name'],
                "DB_Object_Type": data['DB_Object_Type'],
                "Evidence": data['Evidence'],
                "Taxon_ID": data['Taxon_ID'],
                "Aspect": data['Aspect']
            }

            collection.update_one({
                "_id": i}, {
                '$set': json
            }, upsert=True)

            pbar.update(1)

    pbar.close()

    print("\nFinished!")


def add_single_sequence(fasta, collection):

    header, sequence = fasta.id, str(fasta.seq)
    db_id, unique_identifier, entry_name = \
        header.split(' ')[0].split('|')

    prot = {
        "primary_accession": unique_identifier,
        "db": db_id,
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


def load_fasta(filename, collection, start=0):

    print("Loading: %s" % filename)

    collection.create_index("db")
    collection.create_index("entry_name")
    collection.create_index("primary_accession")

    n = count_lines(filename, sep=bytes('>', 'utf8'))
    pbar = tqdm(range(n), desc="sequences loaded")

    with open(filename, 'r') as f:

        fasta_sequences = parse_fasta(f, 'fasta')

        for i, seq in enumerate(fasta_sequences):

            if i < start:
                pbar.update(1)
                continue

            add_single_sequence(seq, collection)

            pbar.update(1)

    pbar.close()

    print("\nFinished!")


def maybe_download_and_unzip(url, data_dir, fname):
    if not os.path.exists("%s/%s.gz" % (data_dir, fname)):
        wget(url, fname, data_dir)
    if not os.path.exists("%s/%s" % (data_dir, fname)):
        unzip(data_dir, fname)


if __name__ == "__main__":

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    prefix, fname = None, None

    if args.collection == "uniprot":

        prefix = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
        fname = "uniprot_trembl.fasta"
        maybe_download_and_unzip(prefix, data_dir, fname)
        load_fasta("%s/%s" % (data_dir, fname), db[args.collection])

    elif args.collection == "sprot":

        prefix = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
        fname = "uniprot_sprot.fasta"
        maybe_download_and_unzip(prefix, data_dir, fname)
        load_fasta("%s/%s" % (data_dir, fname), db[args.collection])

    elif args.collection == "goa_uniprot":

        prefix = "http://geneontology.org/gene-associations/"
        fname = "goa_uniprot_all.gaf"
        maybe_download_and_unzip(prefix, data_dir, fname)
        load_gaf("%s/%s" % (data_dir, fname), db[args.collection])

    elif args.collection == "goa_pdb":

        prefix = "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/PDB"
        fname = "goa_pdb.gaf"
        maybe_download_and_unzip(prefix, data_dir, fname)
        load_gaf("%s/%s" % (data_dir, fname), db[args.collection])

    else:
        print("Unrecognised data source: %s" % args.src)
        exit(0)


