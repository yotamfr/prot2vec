import sys
import datetime
from Bio.UniProt import GOA
from pymongo import MongoClient

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True,
                    help="Give full path to Gene Ontology GAF file")
parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                    help="Supply the URL of MongoDB")
parser.add_argument("--collection", type=str, choices=['goa_pdb', 'goa_uniprot'],
                    default="goa_uniprot", help="Give collection name.")
parser.add_argument('--exp', action='store_true', default=False,
                    help="Load only experimentally validated annotations.")
args = parser.parse_args()

client = MongoClient(args.mongo_url)
db_name = 'prot2vec'
collection = client[db_name][args.collection]


Experimental_Codes = {

    "ECO:0000269": "EXP",
    "ECO:0000314": "IDA",

    "ECO:0000353": "IPI",
    "ECO:0000315": "IMP",

    "ECO:0000316": "IGI",
    "ECO:0000270": "IEP"
}
exp_codes = set(Experimental_Codes.values())


def load_gaf(filename, start=collection.count({})):   # load GOA in a flat structure

    print("Loading %s" % filename)

    collection.create_index("DB_Object_ID")
    collection.create_index("DB")
    collection.create_index("GO_ID")
    collection.create_index("Evidence")
    collection.create_index("Aspect")
    collection.create_index("Date")

    with open(filename, 'r') as handler:

        goa_iterator = GOA.gafiterator(handler)

        for i, data in enumerate(goa_iterator):

            if i % 100 == 0:
                sys.stdout.write("\rProcessed annotations\t%s" % i)

            if i < start or (args.exp and data['Evidence'] not in exp_codes):
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

    print("\nFinished!")


if __name__ == "__main__":
    load_gaf(args.input, 0)
