#!/bin/bash

if [ -z "$1" ]
  then
    echo "No data directory supplied"
    exit
fi

DATA=$1
mkdir -p "$DATA"


URL_UNIPROT_KB="ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
FNAME_TREMBL="uniprot_trembl.fasta.gz"
URL_GOA_UNIPROT="ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT"
FNAME_GOA_UNIPROT="goa_uniprot_all.gpa.gz"

function wgetAndGunzip () {
   url=$1
   fname=$2
   data=$DATA

   rm -f "$data/$fname"
   wget -O "$data/$fname" "$url/$fname"
   gunzip "$data/$fname"
}


wgetAndGunzip "$URL_GOA_UNIPROT" "$FNAME_GOA_UNIPROT"

wgetAndGunzip "$URL_UNIPROT_KB" "$FNAME_TREMBL"



