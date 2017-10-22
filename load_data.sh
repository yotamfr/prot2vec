#!/bin/bash

if [ -z "$1" ]
  then
    echo "No data directory supplied"
    exit
fi

DATA=$1
mkdir -p "$DATA"


#URL_FASTA="ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
#FASTA_FNAME="uniprot_trembl.fasta"

URL_FASTA="ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
FASTA_FNAME="uniprot_sprot.fasta"

#URL_GOA="ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT"
#GAF_FNAME="goa_uniprot_all.gaf"

URL_GOA="ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/PDB"
GAF_FNAME="goa_pdb.gaf"


function cleanup()
{
   fpath=$1

   rm -f "$fpath.gz"
   rm -f "$fpath"
}


function wget_and_gunzip ()
{
   url=$1
   fname=$2
   data=$DATA

   echo "Starting download"
   wget -O "$data/$fname.gz" "$url/$fname.gz"

   echo "Unzipping..."
   gunzip -f "$data/$fname.gz"
}


echo "Restarting mongodb"
sudo service mongod restart

cleanup "$DATA/$GAF_FNAME"
wget_and_gunzip "$URL_GOA" "$GAF_FNAME"
src/python/load_annotations.py --fasta "$DATA/$GAF_FNAME" --collection goa_pdb
cleanup "$DATA/$GAF_FNAME"

cleanup "$DATA/$FASTA_FNAME"
wget_and_gunzip "$URL_FASTA" "$FASTA_FNAME"
src/python/load_sequences.py --gaf "$DATA/$FASTA_FNAME" --collection sprot
cleanup "$DATA/$FASTA_FNAME"
