import datetime
import itertools
from xml.etree.ElementTree import fromstring

import requests
from Bio import SeqIO
from Bio.UniProt import GOA
from parse import parse
from pymongo import MongoClient
from tqdm import tqdm
from xmljson import badgerfish as bf

import utils
import parameters as params
from models import EcodDomain

args = params.arguments
logger = utils.get_logger("mongoscripts")

client = MongoClient('mongodb://localhost:27017/')
dbname = args["db"]
db = client[dbname]


def parse_int(x):
    try:
        return int(x)
    except ValueError:
        return "NA"

def parse_float(x):
    try:
        return float(x)
    except ValueError:
        return "NA"


def parse_list(x):
    return x.strip().split(' | ')


def parse_bool(x):
    if x == "yes":
        return True
    elif x == "no":
        return False


def get_GO_terms(pdb_id):
    pdb, chain = pdb_id[:4], pdb_id[4:]
    req = requests.get('http://www.rcsb.org/pdb/rest/goTerms?structureId=%s.%s' % (pdb, chain))
    if req.status_code != 200:   # then assume it's a .cif
        raise requests.HTTPError('HTTP Error %s' % req.status_code)
    data = bf.data(fromstring(req.content))['goTerms']
    return [] if 'term' not in data else data['term']


def load_ecod_sequences(start=db.ecod.count({})):

    numseq = 0
    filename = args["ecod_fasta"]
    logger.info("Countig ECOD sequences.")
    fasta_sequences = SeqIO.parse(open(filename), 'fasta')
    for fasta in fasta_sequences: numseq += 1
    logger.info("Loading %s ECOD sequences to %s ..." % (numseq, dbname))
    fasta_sequences = SeqIO.parse(open(filename), 'fasta')
    for i in tqdm(range(numseq), desc="sequences processed"):
        fasta = next(fasta_sequences)
        ecod = EcodDomain(header=fasta.description, sequence=fasta.seq)
        db.ecod.update_one({"_id": ecod.eid}, {
            "$set": {
                "uid": ecod.uid,
                "complex": ecod.pdb,
                "chain": ecod.chain,
                "num": int(ecod.num),
                "ecod_id": ecod.eid,
                "sequence": str(ecod.seq),
                "hierarchy": ecod.hierarchy,
                "loci": [{"chain": loc.chain, "start": loc.start, "end": loc.end} for loc in ecod.loci],
            }
        }, upsert=True)

    logger.info("\nFinished!")


def load_pdb_sequences(collection, filename, start=None, fetch_go=False):

    numseq = 0
    if not start: start = collection.count({}) + 5
    logger.info("Countig PDB sequences.")
    fasta_sequences = SeqIO.parse(open(filename), 'fasta')
    for _ in fasta_sequences: numseq += 1
    logger.info("Loading %s PDB sequences to %s ..." % (numseq, dbname))
    fasta_sequences = SeqIO.parse(open(filename), 'fasta')

    formatC = "{pdb_id}{:s}{:w}{:s}{seq_length:INT}{:s}{method}{:s}{resolution:FLOAT}{:s}{r_val_f:FLOAT}" \
              "{:s}{r_val_w:FLOAT} yes{desc}<{uniprot_str}>{:s}[{organism}]"
    formatD = "{pdb_id}{:s}{:w}{:s}{seq_length:INT}{:s}{method}{:s}{resolution:FLOAT}{:s}{r_val_f:FLOAT}" \
              "{:s}{r_val_w:FLOAT} no{desc}<{uniprot_str}>{:s}[{organism}]"

    uniprot_format = "{uid}({start:INT}-{end:INT})"

    for i in tqdm(range(numseq), desc="sequences processed"):

        fasta = next(fasta_sequences)

        if i < start: continue
        d = None

        if ' ||' in fasta.description:
            desc, dup = fasta.description.split(' ||')
        else:
            desc, dup = fasta.description, None

        if not d: d = parse(formatC, desc,
                            dict(INT=parse_int, FLOAT=parse_float, BOOL=parse_bool, LIST=parse_list))
        if not d: d = parse(formatD, desc,
                            dict(INT=parse_int, FLOAT=parse_float, BOOL=parse_bool, LIST=parse_list))
        if not d: continue

        descriptors = d["desc"].strip().split(' | ')

        uniprot = None if d["uniprot_str"] == "NA" else \
            [parse(uniprot_format, u, dict(INT=parse_int)) if '(' in u
             else {"uid": u, "start": -1, "end": -1} for u in d["uniprot_str"].split(' | ')]

        assert d["pdb_id"] == fasta.id

        terms = [] if not fetch_go else get_GO_terms(fasta.id)

        collection.update_one({"_id": fasta.id}, {
            "$set": {
                "pdb_id": d["pdb_id"],
                "complex": d["pdb_id"][:4],
                "chain": d["pdb_id"][4:],
                "sequence": str(fasta.seq),
                "seq_length": d["seq_length"],
                "method": d["method"],
                "resolution": d["resolution"],
                "r_val_free": d["r_val_f"],
                "r_val_work": d["r_val_w"],
                "uniprot": [] if not uniprot
                else [{
                     "uid": u["uid"],
                     "start": u["start"],
                     "end": u["end"]
                 } for u in uniprot],
                "organism": d["organism"],
                "goTerms":
                    [{"goid": t['@id'],
                      "ontology": t['detail']['@ontology'],
                      "name": t['detail']['@name'],
                      "definition": t['detail']['@definition']
                      } for t in terms],
                "descriptors": descriptors,
                "duplicates": [] if not dup else dup.split(' ')
            }
        }, upsert=True)

    logger.info("\nFinished!")


def load_pdb_goa(start=db.goa.count({})): # load GOA in a flat structure

    logger.info("Countig GeneOntology annotations ...")
    numannots = 0
    filename = args["pdb_gaf"]

    with open(filename, 'r') as handler:
        goa = GOA.gafiterator(handler)
        for line in goa:
            numannots += 1
    logger.info("Loading %s GO annotations..." % numannots)
    with open(filename, 'r') as handler:
        goa = GOA.gafiterator(handler)
        for i in tqdm(range(numannots), desc="annotations already processed"):

            data = next(goa)

            if i < start: continue

            date = datetime.datetime.strptime(data['Date'], "%Y%m%d").date()
            assert data["DB_Object_ID"] == data["DB_Object_Symbol"]

            pdb = data["DB_Object_ID"][:4]
            chain = data["DB_Object_ID"][5:]
            json = {
                "PDB_ID":  pdb+chain,
                "Entry_ID": pdb,
                "Chain": chain,
                "DB_Object_ID": data['DB_Object_ID'],
                "With": data['With'],
                "Assigned_By": data["Assigned_By"],
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
            db.goa.update_one( {
                "_id": i}, {
                '$set': json
            }, upsert=True)

    logger.info("\nFinished!")


def add_single_uniprot(fasta):

    header, sequence = fasta.id, str(fasta.seq)
    dbname, UniqueIdentifier, EntryName = header.split(' ')[0].split('|')

    prot = {
        "primary_accession": UniqueIdentifier,
        "db": dbname,
        "entry_name": EntryName,
        "sequence": sequence,
        "length": len(sequence),
        "created_at": datetime.datetime.utcnow(),
        "header": header
    }

    db.uniprot.update_one({
        "_id": UniqueIdentifier}, {
        "$set": prot
    }, upsert=True)


def load_uniprot(start=db.uniprot.count({})):   # http://www.uniprot.org/help/fasta-headers
    numseq = 0
    logger.info("Countig Uniprot sequences.")
    fasta_sequences = SeqIO.parse(open(args["uniprot_fasta"]),'fasta')
    for fasta in fasta_sequences: numseq += 1
    logger.info("\nLoading %s Uniprot sequences to cafa3 ...\n" % numseq)
    fasta_sequences = SeqIO.parse(open(args["uniprot_fasta"]), 'fasta')
    for i in tqdm(range(numseq), desc="sequences already processed"):
        if i < start: continue
        add_single_uniprot(next(fasta_sequences))
    logger.info("\nFinished!")


# def read_clstr(collection, cluster_filename):
#     # parse through the .clstr file and create a dictionary
#     # with the sequences per cluster
#
#     numseq = 0
#
#     # open the cluster file and set the output dictionary
#     cluster_file, cluster_dic = open(cluster_filename), {}
#
#     logger.info("Reading cluster groups...")
#     # parse through the cluster file and store the cluster name + sequences in the dictionary
#     cluster_groups = (x[1] for x in itertools.groupby(cluster_file, key=lambda line: line[0] == '>'))
#     for cluster in cluster_groups:
#         name = int(next(cluster).strip().split()[-1])
#         seqs = [seq.split('>')[1].split('...')[0].split('|')[1] for seq in next(cluster_groups)]
#         cluster_dic[name] = seqs
#         numseq += len(seqs)
#     logger.info("Writing %s cluster groups..." % len(cluster_dic))
#
#     bar = tqdm(range(numseq), desc="sequences processed")
#
#     for cluster, seqs in cluster_dic.items():
#         for seq in seqs:
#             collection.update_one({"_id": seq}, {
#                 "$set": {
#                     "clstr": cluster
#                 }
#             }, upsert=False)
#         bar.update(len(seqs))
#
#     return cluster_dic


def main():
    load_pdb_sequences(collection=db.pdbnr, filename=args["pdbnr_fasta"])
    # load_ecod_sequences()
    # load_pdb_goa()
    # load_uniprot()


if __name__ == "__main__":
    main()
