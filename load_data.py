import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True,
                    help="Supply working directory for the data")
parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                    help="Supply the URL of MongoDB")
parser.add_argument("--collection", type=str, required=True,
                    choices=['uniprot', 'sprot', 'goa_uniprot', 'goa_pdb'],
                    help="The name of the collection that you want load.")
parser.add_argument('--cleanup', action='store_true', default=False,
                    help="Delete data files after loading is done.")
parser.add_argument('--download', action='store_true', default=False,
                    help="Re-download the data regardless if it's in dir.")
args = parser.parse_args()


def cleanup(datadir, fname):
    os.system("rm -f %s/%s*" % (datadir, fname))


def wget(url, fname, datadir):
    print("Downloading to %s" % datadir)
    os.system('wget -O %s/%s.gz %s/%s.gz' % (datadir, fname, url, fname))


def unzip(datadir, fname):
    print("Unzipping...")
    fpath = '%s/%s' % (datadir, fname)
    os.system('gunzip -c %s.gz > %s' % (fpath, fpath))


if __name__=="__main__":

    datadir, collection = os.path.abspath(args.dir), args.collection

    if collection == "uniprot":

        url = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
        fname = "uniprot_trembl.fasta"
        proc = "load_sequences"

    elif collection == "sprot":

        url = "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
        fname = "uniprot_sprot.fasta"
        proc = "load_sequences"

    elif collection == "goa_uniprot":

        url = "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT"
        fname = "goa_uniprot_all.gaf"
        proc = "load_annotations"

    elif collection == "goa_pdb":

        url = "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/PDB"
        fname = "goa_pdb.gaf"
        proc = "load_annotations"

    else:
        print ("Unrecognised data source: %s" % args.src)
        exit(0)

    if args.download or not os.path.exists("%s/%s.gz" % (datadir, fname)):
        cleanup(datadir, fname)
        wget(url, fname, datadir)
        unzip(datadir, fname)
    elif not os.path.exists("%s/%s" % (datadir, fname)):
        unzip(datadir, fname)

    os.system("python src/python/%s.py --input %s --collection %s --mongo_url %s"
              % (proc, "%s/%s" % (datadir, fname), collection, args.mongo_url))

    if args.cleanup:
        cleanup(datadir, fname)
