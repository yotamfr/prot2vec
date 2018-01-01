import os


from pymongo import MongoClient

client = MongoClient("mongodb://127.0.0.1:27017")
db = client['prot2vec']


UNIPROT20_URL = "http://wwwuser.gwdg.de/%7Ecompbiol/data/hhsuite/databases/hhsuite_dbs/uniprot20_2016_02.tgz"


def prepare_uniprot20():
    if not os.path.exists("dbs/uniprot20_2016_02"):
        # os.system("wget %s --directory-prefix dbs " % UNIPROT20_URL)
        os.system("tar -xvzf dbs/uniprot20_2016_02.tgz")


if __name__ == "__main__":
    prepare_uniprot20()


