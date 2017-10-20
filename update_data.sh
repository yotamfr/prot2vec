#!/bin/bash

if [ -z "$1" ]
  then
    echo "No data directory supplied"
    exit
fi

DATA=$1
mkdir -p "$DATA"


FNAME_GOA_UNIPROT="goa_uniprot_all.gpa.gz"
FNAME_TREMBL="uniprot_trembl.fasta.gz"
URL_GOA_UNIPROT="ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT"
URL_UNIPROT="ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"

rm "$DATA/$FNAME_GOA_UNIPROT"
rm "$DATA/$FNAME_TREMBL"

wget -O "$DATA/$FNAME_GOA_UNIPROT" "$URL_GOA_UNIPROT/$FNAME_GOA_UNIPROT"
wget -O "$DATA/$FNAME_TREMBL" "$URL_UNIPROT/$FNAME_TREMBL"

