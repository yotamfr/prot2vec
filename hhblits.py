import os
import sys
import io

import subprocess
from tqdm import tqdm

import A3MIO
from Bio.Seq import Seq
from Bio import SeqIO, AlignIO
from Bio.SeqRecord import SeqRecord
from Bio.Align.AlignInfo import SummaryInfo
from Bio.Blast.Applications import NcbipsiblastCommandline

from concurrent.futures import ThreadPoolExecutor

from src.python.preprocess import *

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()

import argparse


out_dir = "./hhblits"
if not os.path.exists(out_dir): os.mkdir(out_dir)

prefix_hhsuite = "/usr/share/hhsuite/scripts/"

uniprot20url = "http://wwwuser.gwdg.de/%7Ecompbiol/data/hhsuite/databases/hhsuite_dbs/uniprot20_2016_02.tgz"
uniprot20name = "uniprot20_2016_02"

ASPECT = 'F'
batch_size = 8
num_cpu = 2
GAP = '-'


def prepare_uniprot20():
    if not os.path.exists("dbs/uniprot20_2016_02"):
        # os.system("wget %s --directory-prefix dbs " % UNIPROT20_URL)
        os.system("tar -xvzf dbs/uniprot20_2016_02.tgz")


def _get_annotated_uniprot(db, limit):
    query = {'DB': 'UniProtKB', 'Evidence': {'$in': exp_codes}}

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    seqid2goid, goid2seqid = GoAnnotationCollectionLoader(s, c, ASPECT).load()

    query = {"_id": {"$in": unique(list(seqid2goid.keys())).tolist()}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    return seqid2seq, seqid2goid, goid2seqid


#    CREDIT TO SPIDER2
def read_pssm(pssm_file):
        # this function reads the pssm file given as input, and returns a LEN x 20 matrix of pssm values.

        # index of 'ACDE..' in 'ARNDCQEGHILKMFPSTWYV'(blast order)
        idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)

        # open the two files, read in their data and then close them
        if pssm_file == 'STDIN': fp = sys.stdin
        else: fp = open(pssm_file, 'r')
        lines = fp.readlines()
        fp.close()

        # declare the empty dictionary with each of the entries
        aa = []
        pssm = []

        # iterate over the pssm file and get the needed information out
        for line in lines:
                split_line = line.split()
                # valid lines should have 32 points of data.
                # any line starting with a # is ignored
                try: int(split_line[0])
                except: continue

                if line[0] == '#': continue

                aa_temp = split_line[1]
                aa.append(aa_temp)
                if len(split_line) in (44, 22):
                        pssm_temp = [-float(i) for i in split_line[2:22]]
                elif len(line) > 70:  # in case double digits of pssm
                        pssm_temp = [-float(line[k*3+9: k*3+12]) for k in range(20)]
                        pass
                else: continue
                pssm.append([pssm_temp[k] for k in idx_res])

        return aa, pssm


def _set_unique_ids(input_file, output_file):
    with open(input_file, "rt") as fin:
        with open(output_file, "wt") as fout:
            for j, line in enumerate(fin):
                prefix = line.split()[0]
                if j > 0: line = line.replace(prefix, "0" * (20 - len(str(j))) + "%d" % j)
                fout.write(line)


def _run_parallel_hhblits_msa(sequences):
    records = [SeqRecord(Seq(seq), id) for id, seq in sequences.items()
               if not os.path.exists("%s/%s.out" % (out_dir, id))]
    i, n = 0, len(records)
    pbar = tqdm(range(len(records)), desc="sequences processed")

    while i < n:
        j = min(i+batch_size, n)
        batch = records[i:j]
        sequences_fasta = 'SEQ-%s.fasta' % GoAspect(ASPECT)
        SeqIO.write(batch, open(os.path.join(out_dir, sequences_fasta), 'w+'), "fasta")
        pwd = os.getcwd()
        os.chdir(out_dir)
        cline = "%s/splitfasta.pl %s" % (prefix_hhsuite, sequences_fasta)
        child = subprocess.Popen(cline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=(sys.platform != "win32"))
        handle, _ = child.communicate()
        assert child.returncode == 0

        cline = "%s/multithread.pl \'*.seq\' \'hhblits -i $file -d ../dbs/%s/%s -opsi $name.out -n 2 -cpu %d\'"\
                % (prefix_hhsuite, uniprot20name, uniprot20name, num_cpu)
        child = subprocess.Popen(cline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=(sys.platform != "win32"))

        handle, _ = child.communicate()
        assert child.returncode == 0
        os.chdir(pwd)
        pbar.update(j - i)
        i = j

    pbar.close()


def _hhblits(seq_record):

    global pbar

    seqid, seq = seq_record.id, str(seq_record.seq)
    database = "dbs/uniprot20_2016_02/uniprot20_2016_02"
    cline = "hhblits -i 'stdin' -d %s -n 2 -a3m 'stdout' -o /dev/null" % database
    child = subprocess.Popen(cline,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=(sys.platform != "win32"))
    _, stdout = child.communicate(input=">%s\n%s" % (seqid, seq))
    # cline = NcbipsiblastCommandline(help=True)
    # "psiblast -subject %s.seq -in_msa %s.psi -out_ascii_pssm %s.pssm"
    assert child.returncode == 0
    a3m = list(AlignIO.parse(io.StringIO(stdout), "a3m"))
    pssm = SummaryInfo(a3m[0]).pos_specific_score_matrix(chars_to_ignore=[GAP])
    return seqid, seq, pssm


def _run_parallel_hhblits(sequences, num_threads=4):

    global pbar

    records = [SeqRecord(Seq(v), k) for k, v in sequences.items()]

    pbar = tqdm(range(len(records)), desc="sequences processed")

    e = ThreadPoolExecutor(num_threads)

    for seqid, seq, pssm, in e.map(_hhblits, records):
        db.pssm.update_one({
            "_id": seqid}, {
            '$set': {"pssm": pssm.pssm, "seq": seq}
        }, upsert=True)
        pbar.update(1)


def _run_hhblits_msa(sequences):
    seq_records = [SeqRecord(Seq(seq), id) for id, seq in sequences.items()
                   if not os.path.exists("%s/%s.out" % (out_dir, id))]

    fasta = 'SEQ-%s.fasta' % GoAspect(ASPECT)
    SeqIO.write(seq_records, open(os.path.join(out_dir, fasta), 'w+'), "fasta")
    pwd = os.getcwd()

    os.chdir(out_dir)
    cline = "%s/splitfasta.pl %s" % (prefix_hhsuite, fasta)
    child = subprocess.Popen(cline,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=(sys.platform != "win32"))

    handle, _ = child.communicate()
    assert child.returncode == 0
    os.chdir(pwd)

    pbar = tqdm(range(len(seq_records)), desc="sequences processed")
    for seq in seq_records:
        os.chdir(out_dir)
        cline = "hhblits -i %s.seq -d ../dbs/%s/%s -opsi %s.out -n 2"\
                % (seq.id, uniprot20name, uniprot20name, seq.id)
        child = subprocess.Popen(cline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=(sys.platform != "win32"))

        handle, _ = child.communicate()
        assert child.returncode == 0
        os.chdir(pwd)

        pbar.update(1)

    pbar.close()


# MUST BE RUN AFTER BLITS FINISHED
def _load_pssms(sequences):
    msa_records = [SeqRecord(Seq(seq), id) for id, seq in sequences.items()]
    pbar = tqdm(range(len(msa_records)), desc="sequences processed")

    for seq in msa_records:
        pwd = os.getcwd()
        os.chdir(out_dir)
        # a3m = list(AlignIO.parse(open("%s.a3m" % seq.id, 'r'), "a3m"))
        # pssm = SummaryInfo(a3m[0]).pos_specific_score_matrix(chars_to_ignore=[GAP])
        _set_unique_ids("%s.out" % seq.id, "%s.msa" % seq.id)
        os.system("psiblast -subject %s.seq -in_msa %s.msa -out_ascii_pssm %s.pssm 1>out.log 2>err.log" %
                  (seq.id, seq.id, seq.id))
        aa, pssm = read_pssm("%s.pssm" % seq.id)
        os.chdir(pwd)
        db.pssm.update_one({
            "_id": seq.id}, {
            '$set': {"pssm": pssm, "seq": ''.join(aa)}
        }, upsert=True)
        pbar.update(1)

    pbar.close()


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--limit", type=int, default=None,
                        help="How many sequences for PSSM computation.")
    parser.add_argument('--parallel', action='store_true', default=False,
                        help="Work in parallel mode.")
    parser.add_argument("--num_cpu", type=int, default=2,
                        help="How many cpus for computing PSSM (when running in parallel mode).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="How many sequences in batch (when running in parallel mode).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    num_cpu = args.num_cpu
    batch_size = args.batch_size

    prepare_uniprot20()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    lim = args.limit

    seqs, _, _ = _get_annotated_uniprot(db, lim)
    # if args.parallel:
    #     _run_parallel_hhblits_msa(seqs)
    # else:
    #     _run_hhblits_msa(seqs)
    # _load_pssms(seqs)

    _run_parallel_hhblits(seqs, num_cpu)
