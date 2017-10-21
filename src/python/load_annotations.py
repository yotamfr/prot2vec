import sys
import datetime
from Bio.UniProt import GOA
from pymongo import MongoClient
from tqdm import tqdm

import argparse

client = MongoClient('mongodb://localhost:27017/')
dbname = 'prot2vec'
db = client[dbname]


def load_gaf(filename, start=db.goa.count({})):   # load GOA in a flat structure

    print("Loading %s" % filename)

    num_annots = 0
    with open(filename, 'r') as handler:
        goa_iterator = GOA.gafiterator(handler)
        for _ in goa_iterator:
            sys.stdout.write("\rCounting annotations\t%s" % num_annots)
            num_annots += 1

    db.goa.create_index("DB_Object_ID")
    db.goa.create_index("DB")
    db.goa.create_index("GO_ID")
    db.goa.create_index("Evidence")
    db.goa.create_index("Aspect")
    db.goa.create_index("Date")

    print("Loading %s GO annotations..." % num_annots)

    with open(filename, 'r') as handler:
        goa_iterator = GOA.gafiterator(handler)
        for i in tqdm(range(num_annots), desc="annotations already processed"):
            data = next(goa_iterator)
            if i < start:
                continue
            date = datetime.datetime.strptime(data['Date'], "%Y%m%d").date()
            assert data["DB_Object_ID"] == data["DB_Object_Symbol"]

            json = {
                "DB_Object_ID": data['DB_Object_ID'],
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
            db.goa.update_one({
                "_id": i}, {
                '$set': json
            }, upsert=True)

    print("\nFinished!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gaf", type=str, help="Give full path to Gene Ontology GAF file")
    args = parser.parse_args()
    load_gaf(args.gaf, 0)

if __name__ == "__main__":
    main()
