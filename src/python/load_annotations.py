#!python3

import sys
import datetime
from Bio.UniProt import GOA
from pymongo import MongoClient

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gaf", type=str, help="Give full path to Gene Ontology GAF file")
parser.add_argument("--collection", type=str, choices=['goa_pdb', 'goa_uniprot'],
                    default="goa_uniprot", help="Give collection name.")

args = parser.parse_args()

client = MongoClient('mongodb://localhost:27017/')
db_name = 'prot2vec'
collection = client[db_name][args.collection]


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

            if i < start: continue

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


def main():
    load_gaf(args.gaf, 0)


if __name__ == "__main__":
    main()
