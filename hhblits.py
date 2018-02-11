import os
import sys
import io
import glob

from datetime import datetime, timedelta

from numpy import unique

import subprocess
from tqdm import tqdm

from Bio.Seq import Seq
from Bio import SeqIO, AlignIO
from Bio.SeqRecord import SeqRecord
from Bio.Align.AlignInfo import SummaryInfo

from concurrent.futures import ThreadPoolExecutor

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()

from src.python.consts import *

import argparse


out_dir = "./hhblits"

if not os.path.exists(out_dir): os.mkdir(out_dir)

prefix_hhsuite = "/usr/share/hhsuite"
prefix_blast = "/usr/bin"
uniprot20url = "http://wwwuser.gwdg.de/%7Ecompbiol/data/hhsuite/databases/hhsuite_dbs/uniprot20_2016_02.tgz"
uniprot20name = "uniprot20_2016_02"
# uniprot20name = "uniprot20_2013_03"

batch_size = 2
num_cpu = 2
max_filter = 20000
coverage = 0
mact = 0.9


IGNORE = [aa for aa in map(str.lower, AA.aa2index.keys())] + ['-']  # ignore deletions + insertions


def prepare_uniprot20():
    if not os.path.exists("dbs/uniprot20_2016_02"):
        # os.system("wget %s --directory-prefix dbs " % UNIPROT20_URL)
        os.system("tar -xvzf dbs/uniprot20_2016_02.tgz")


def _get_annotated_uniprot(db, limit, min_length=1, max_length=3000):
    q = {'DB': 'UniProtKB', 'Evidence': {'$in': exp_codes}}
    s = db.goa_uniprot.find(q)
    if limit: s = s.limit(limit)
    uniprot_ids = list(map(lambda doc: doc["DB_Object_ID"], s))

    q = {"_id": {"$in": unique(uniprot_ids).tolist()}, "length": {"$gte": min_length, "$lte": max_length}}
    s = db.uniprot.find(q)
    seqid2seq = {doc["_id"]: doc["sequence"] for doc in s}

    return sorted(((k, v) for k, v in seqid2seq.items()), key=lambda pair: -len(pair[1]))


#    CREDIT TO SPIDER2
def read_pssm(pssm_file):
        # this function reads the pssm file given as input, and returns a LEN x 20 matrix of pssm values.

        # index of 'ACDE..' in 'ARNDCQEGHILKMFPSTWYV'(blast order)
        # idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)

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
                        pssm_temp = [float(i) for i in split_line[2:22]]
                elif len(line) > 70:  # in case double digits of pssm
                        pssm_temp = [float(line[k*3+9: k*3+12]) for k in range(20)]
                        pass
                else: continue
                pssm.append({AA.index2aa[k]: pssm_temp[k] for k in range(20)})

        return aa, pssm


def _set_unique_ids(input_file, output_file):
    with open(input_file, "rt") as fin:
        with open(output_file, "wt") as fout:
            for j, line in enumerate(fin):
                prefix = line.split()[0]
                if j > 0: line = line.replace(prefix, "0" * (20 - len(str(j))) + "%d" % j)
                fout.write(line)


def _run_hhblits_batched(sequences):

    is_new = {"$gte": datetime.utcnow() - timedelta(days=5)}
    records = [SeqRecord(Seq(seq), seqid) for (seqid, seq) in sequences]

    i, n = 0, len(records)
    pbar = tqdm(range(len(records)), desc="sequences processed")

    while i < n:
        batch = []
        while len(batch) < batch_size:
            if len(records) > 0:
                seq = records.pop()
                pbar.update(1)
                i += 1
            else:
                break
            if db.pssm.find_one({"_id": seq.id, "updated_at": is_new, "pssm": {"$exists": True}}):
                continue

            db.pssm.update_one({
                "_id": seq.id}, {
                '$set': {"updated_at": datetime.utcnow()}
            }, upsert=True)

            batch.append(seq)

        pwd = os.getcwd()
        os.chdir(out_dir)

        if len(glob.glob('*.seq')): os.system("rm *.seq")

        sequences_fasta = 'batch-%d.fasta' % (i//batch_size)
        SeqIO.write(batch, open(sequences_fasta, 'w+'), "fasta")
        cline = "%s/scripts/splitfasta.pl %s 1>/dev/null 2>&1" \
                % (prefix_hhsuite, sequences_fasta)
        assert os.WEXITSTATUS(os.system(cline)) == 0

        if len(glob.glob('*.a3m')): os.system("rm *.a3m")

        hhblits_cmd = "%s/bin/hhblits -i $file -d ../dbs/%s/%s -oa3m $name.a3m -n 2 -maxfilt %d -mact %s"\
                      % (prefix_hhsuite, uniprot20name, uniprot20name, max_filter, mact)
        cline = "%s/scripts/multithread.pl \'*.seq\' \'%s\' -cpu %d 1>/dev/null 2>&1" \
                % (prefix_hhsuite, hhblits_cmd, num_cpu)
        assert os.WEXITSTATUS(os.system(cline)) == 0

        if coverage > 0:
            if len(glob.glob('*.fil')): os.system("rm *.fil")
            hhfilter_cmd = "%s/bin/hhfilter -i $file -o $name.fil -cov %d" \
                           % (prefix_hhsuite, coverage)
            cline = "%s/scripts/multithread.pl \'*.a3m\' \'%s\' -cpu %d 1>/dev/null 2>&1" \
                    % (prefix_hhsuite, hhfilter_cmd, num_cpu)
            assert os.WEXITSTATUS(os.system(cline)) == 0

        suffix = 'fil' if coverage > 0 else 'a3m'

        if output_fasta:
            reformat_cmd = "%s/scripts/reformat.pl -r a3m fas $file $name.fas" % prefix_hhsuite
            cline = "%s/scripts/multithread.pl \'*.%s\' \'%s\' -cpu %d 1>/dev/null 2>&1" \
                    % (prefix_hhsuite, suffix, reformat_cmd, num_cpu)
            assert os.WEXITSTATUS(os.system(cline)) == 0

        reformat_cmd = "%s/scripts/reformat.pl -r a3m psi $file $name.psi" % prefix_hhsuite
        cline = "%s/scripts/multithread.pl \'*.%s\' \'%s\' -cpu %d 1>/dev/null 2>&1"\
                % (prefix_hhsuite, suffix, reformat_cmd, num_cpu)
        assert os.WEXITSTATUS(os.system(cline)) == 0

        e = ThreadPoolExecutor(num_cpu)
        for (seq, pssm, aln) in e.map(_get_pssm, [seq for seq in batch if os.path.exists("%s.a3m" % seq.id)]):
            db.pssm.update_one({
                "_id": seq.id}, {
                '$set': {"pssm": pssm,
                         "alignment": aln,
                         "seq": str(seq.seq),
                         "length": len(seq.seq),
                         "updated_at": datetime.utcnow()}
            }, upsert=True)

        if cleanup: os.system("rm ./*")

        os.chdir(pwd)

    pbar.close()


def _read_a3m(seq):
    return seq, open("%s.a3m" % str(seq.id), 'r').read()


def _run_hhblits_parallel(sequences):

    global pbar

    pwd = os.getcwd()
    os.chdir(out_dir)
    is_new = {"$gte": datetime.utcnow() - timedelta(days=5)}
    records = [SeqRecord(Seq(seq), seqid) for (seqid, seq) in sequences]

    n = len(records)
    pbar = tqdm(range(n), desc="sequences processed")
    e = ThreadPoolExecutor(num_cpu)

    for (seq, pssm, aln) in e.map(_hhblits, [seq for seq in records if os.path.exists("%s.a3m" % seq.id)]):
        db.pssm.update_one({
            "_id": seq.id}, {
            '$set': {"pssm": pssm,
                     "alignment": aln,
                     "seq": str(seq.seq),
                     "length": len(seq.seq),
                     "updated_at": datetime.utcnow()}
        }, upsert=True)

    if cleanup: os.system("rm ./*")
    os.chdir(pwd)
    pbar.update(1)


def _hhblits(seq_record):

    global pbar

    seqid, seq = seq_record.id, str(seq_record.seq)
    database = "../dbs/uniprot20_2016_02/uniprot20_2016_02"
    SeqIO.write([seq_record], open("%s.seq" % seqid, 'w+'), "fasta")

    cline = "%s/bin/hhblits -i %s.seq -d %s -n 2 -oa3m %s.a3m 1>/dev/null 2>&1" % (prefix_hhsuite, seqid, database, seqid)
    assert os.WEXITSTATUS(os.system(cline)) == 0

    if output_fasta:
        cline = "%s/scripts/reformat.pl -r a3m fas %s.a3m %s.fas 1>/dev/null 2>&1" % (prefix_hhsuite, seqid, seqid)
        assert os.WEXITSTATUS(os.system(cline)) == 0

    cline = "%s/scripts/reformat.pl -r a3m psi %s.a3m %s.psi 1>/dev/null 2>&1" % (prefix_hhsuite, seqid, seqid)
    assert os.WEXITSTATUS(os.system(cline)) == 0

    _set_unique_ids("%s.psi" % seqid, "%s.msa" % seqid)

    aln = []
    with open("%s.psi" % seqid, 'rt') as f:
        for line in f.readlines():
            r = line.strip().split()
            if len(r) < 2:
                continue
            aln.append(r)

    cline = "%s/psiblast -subject %s.seq -in_msa %s.msa -out_ascii_pssm %s.pssm 1>/dev/null 2>&1" \
            % (prefix_blast, seqid, seqid, seqid)
    assert os.WEXITSTATUS(os.system(cline)) == 0

    # aln = list(AlignIO.parse(open("%s.fas" % seq.id, 'r'), "fasta"))
    # pssm = SummaryInfo(aln[0]).pos_specific_score_matrix(chars_to_ignore=IGNORE)
    aa, pssm = read_pssm("%s.pssm" % seqid)

    if cleanup: os.remove("%s.*" % seqid)

    return seq_record, pssm, aln


# MUST BE RUN AFTER HHBLITS FINISHED
def _get_pssm(seq):

    # cline = "%s/scripts/addss.pl %s.a3m" % (prefix_hhsuite, seq.id)
    # assert os.WEXITSTATUS(os.system(cline)) == 0

    # cline = "%s/scripts/reformat.pl -r a3m psi %s.a3m %s.psi" \
    #         % (prefix_hhsuite, seq.id, seq.id)
    # assert os.WEXITSTATUS(os.system(cline)) == 0

    _set_unique_ids("%s.psi" % seq.id, "%s.msa" % seq.id)

    aln = []
    with open("%s.psi" % seq.id, 'rt') as f:
        for line in f.readlines():
            r = line.strip().split()
            if len(r) < 2:
                continue
            aln.append(r)

    cline = "%s/psiblast -subject %s.seq -in_msa %s.msa -out_ascii_pssm %s.pssm 1>/dev/null 2>&1" \
            % (prefix_blast, seq.id, seq.id, seq.id)
    assert os.WEXITSTATUS(os.system(cline)) == 0

    # aln = list(AlignIO.parse(open("%s.fas" % seq.id, 'r'), "fasta"))
    # pssm = SummaryInfo(aln[0]).pos_specific_score_matrix(chars_to_ignore=IGNORE)
    aa, pssm = read_pssm("%s.pssm" % seq.id)

    return seq, pssm, aln


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--prefix_hhsuite", type=str, default='/usr/share/hhsuite',
                        help="Specify where you installed hhsuite.")
    parser.add_argument("--prefix_blast", type=str, default='/usr/bin',
                        help="Specify where you installed ncbi blast.")
    parser.add_argument("--limit", type=int, default=None,
                        help="How many sequences for PSSM computation.")
    parser.add_argument("--max_filter", type=int, default=20000,
                        help="How many sequences to include in the MSA for PSSM computation.")
    parser.add_argument("--num_cpu", type=int, default=2,
                        help="How many cpus for computing PSSM (when running in batched mode).")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="How many sequences in batch (when running in parallel mode).")
    parser.add_argument("--coverage", type=int, default=0,
                        help="The desired coverage (for the alignment algorithm).")
    parser.add_argument("--mact", type=int, default=0.9,
                        help="Set the Max ACC (mact) threshold (for the alignment algorithm).")
    parser.add_argument('--keep_files', action='store_true', default=False,
                        help="Whether to keep intermediate files.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    num_cpu = args.num_cpu
    batch_size = args.batch_size
    max_filter = args.max_filter
    coverage = args.coverage
    mact = args.mact
    prefix_blast = args.prefix_blast
    prefix_hhsuite = args.prefix_hhsuite

    cleanup = not args.keep_files
    output_fasta = args.keep_files

    os.environ['HHLIB'] = prefix_hhsuite

    prepare_uniprot20()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    lim = args.limit

    db.pssm.create_index("updated_at")

    seqs = _get_annotated_uniprot(db, lim)

    _run_hhblits_batched(seqs)
