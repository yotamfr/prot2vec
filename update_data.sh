#!/bin/bash

if [ -z "$1" ]
  then
    echo "No data directory supplied"
    exit
fi

DATA=$1
mkdir -p "$DATA"


URL_FASTA="ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
FASTA_FNAME="uniprot_trembl.fasta.gz"

URL_GOA_UNIPROT="ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT"
GAF_FNAME="goa_uniprot_all.gaf.gz"

#URL_GOA_PDB="ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/PDB"
#GAF_FNAME="goa_pdb.gaf.gz"


function wget_and_gunzip () {
   url=$1
   fname=$2
   data=$DATA

   rm -f "$data/$fname"
   wget -O "$data/$fname" "$url/$fname"
   gunzip -f "$data/$fname"
}

wget_and_gunzip "$URL_GOA_UNIPROT" "$GAF_FNAME"
src/python/load_annotations.py "../../$DATA/$GAF_FNAME"


#wget_and_gunzip "$URL_FASTA" "$FASTA_FNAME"
#src/python/load_sequences.py "../../$DATA/$FASTA_FNAME"




