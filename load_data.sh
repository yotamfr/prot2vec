#!/bin/bash

if [ -z "$1" ]
  then
    echo "No data directory supplied"
    exit
fi

DATA=$1
mkdir -p "$DATA"


URL_FASTA="ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
FASTA_FNAME="uniprot_trembl.fasta"

#URL_FASTA="ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
#FASTA_FNAME="uniprot_sprot.fasta"

URL_GOA="ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT"
GAF_FNAME="goa_uniprot_all.gaf"

#URL_GOA="ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/PDB"
#GAF_FNAME="goa_pdb.gaf"


function wget_and_gunzip () {
   url=$1
   fname=$2
   data=$DATA

   rm -f "$data/$fname.gz"
   rm -f "$data/$fname"

   echo "Starting download"
   wget -O "$data/$fname.gz" "$url/$fname.gz"

   echo "Unzipping..."
   gunzip -f "$data/$fname.gz"
}

echo "Restarting mongodb"
sudo service mongod restart

wget_and_gunzip "$URL_GOA" "$GAF_FNAME"
dist/load_annotations/load_annotations "$DATA/$GAF_FNAME"

wget_and_gunzip "$URL_FASTA" "$FASTA_FNAME"
dist/load_sequences/load_sequences "$DATA/$FASTA_FNAME"
