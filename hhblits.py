import os
import sys
import io

import subprocess
from tqdm import tqdm

from Bio.Seq import Seq
from Bio import SeqIO, AlignIO
from Bio.SeqRecord import SeqRecord
from Bio.Align.AlignInfo import SummaryInfo

from concurrent.futures import ThreadPoolExecutor

from src.python.preprocess import *

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()

import argparse


out_dir = "./hhblits"
if not os.path.exists(out_dir): os.mkdir(out_dir)

prefix_hhsuite = "/usr/share/hhsuite/scripts"
uniprot20url = "http://wwwuser.gwdg.de/%7Ecompbiol/data/hhsuite/databases/hhsuite_dbs/uniprot20_2016_02.tgz"
uniprot20name = "uniprot20_2016_02"

batch_size = 8
num_cpu = 2
max_filter = 2000
IGNORE = [aa for aa in map(str.lower, AA.dictionary.keys())] + ['-']  # ignore deletions + insertions


def prepare_uniprot20():
    if not os.path.exists("dbs/uniprot20_2016_02"):
        # os.system("wget %s --directory-prefix dbs " % UNIPROT20_URL)
        os.system("tar -xvzf dbs/uniprot20_2016_02.tgz")


def _get_annotated_uniprot(db, limit, max_length=2818):
    query = {'DB': 'UniProtKB', 'Evidence': {'$in': exp_codes}}

    c = limit if limit else db.goa_uniprot.count(query)
    s = db.goa_uniprot.find(query)
    if limit: s = s.limit(limit)

    uniprot_ids = []
    for asp in ['F', 'C', 'P']:
        seqid2goid, _ = GoAnnotationCollectionLoader(s, c, asp).load()
        uniprot_ids.extend(list(seqid2goid.keys()))

    query = {"_id": {"$in": unique(uniprot_ids).tolist()}, "length": {"$lte": max_length}}
    num_seq = db.uniprot.count(query)
    src_seq = db.uniprot.find(query)

    seqid2seq = UniprotCollectionLoader(src_seq, num_seq).load()

    return sorted(((k, v) for k, v in seqid2seq.items()), key=lambda pair: len(pair[1]))


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


def _run_hhblits_batched(sequences, cleanup=False):
    os.environ['HHLIB'] = "/usr/share/hhsuite"

    records = [SeqRecord(Seq(seq), seqid) for (seqid, seq) in sequences]
    i, n = 0, len(records)
    pbar = tqdm(range(len(records)), desc="sequences processed")

    while i < n:
        j = min(i+batch_size, n)
        batch = records[i:j]
        pwd = os.getcwd()
        os.chdir(out_dir)

        sequences_fasta = 'batch-%d.fasta' % (j//batch_size)
        SeqIO.write(batch, open(sequences_fasta, 'w+'), "fasta")
        cline = "%s/splitfasta.pl %s" % (prefix_hhsuite, sequences_fasta)

        child = subprocess.Popen(cline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=(sys.platform != "win32"))

        handle, _ = child.communicate()
        assert child.returncode == 0

        hhblits_cmd = "hhblits -i $file -d ../dbs/%s/%s -oa3m $name.a3m -n 2 -maxfilt %d -mact 0.9 -cpu %d" % \
                      (uniprot20name, uniprot20name, max_filter, num_cpu)
        cline = "%s/multithread.pl \'*.seq\' \'%s\'" % (prefix_hhsuite, hhblits_cmd)
        child = subprocess.Popen(cline,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=(sys.platform != "win32"))

        handle, _ = child.communicate()
        assert child.returncode == 0

        # e = ThreadPoolExecutor(num_cpu)
        # for (seq, pssm, aa) in e.map(_get_pssm, batch):
        #     db.pssm.update_one({
        #         "_id": seq.id}, {
        #         '$set': {"pssm": pssm,
        #                  "seq": str(seq.seq),
        #                  "length": len(seq.seq)}
        #     }, upsert=True)

        if cleanup:
            os.system("rm ./*")
        os.chdir(pwd)

        pbar.update(j - i)
        i = j

    pbar.close()


def _read_a3m(seq):
    return seq, open("%s.a3m" % str(seq.id), 'r').read()


def _hhblits(seq_record, cleanup=True):

    global pbar

    seqid, seq = seq_record.id, str(seq_record.seq)
    database = "dbs/uniprot20_2016_02/uniprot20_2016_02"
    msa_pth = os.path.join(tmp_dir, "%s.msa" % seqid)

    # cline = "hhblits -i 'stdin' -d %s -n 2 -opsi %s -o /dev/null" % (database, msa_pth)

    cline = "hhblits_omp -i 'stdin' -d %s -n 2 -oa3m 'stdout'" % database,
    child = subprocess.Popen(cline,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=(sys.platform != "win32"))
    _, stdout = child.communicate(input=">%s\n%s" % (seqid, seq))
    assert child.returncode == 0

    sys.stdin = io.StringIO(stdout)

    seq_pth = os.path.join(tmp_dir, "%s.seq" % seqid)
    SeqIO.write(seq_record, open(seq_pth, 'w+'), "fasta")
    mat_pth = os.path.join(tmp_dir, "%s.pssm" % seqid)

    cline = "psiblast -subject %s -in_msa %s -out_ascii_pssm %s" % (seq_pth, msa_pth, mat_pth)
    child = subprocess.Popen(cline,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=(sys.platform != "win32"))
    _, _ = child.communicate()
    assert child.returncode == 0

    aa, pssm = read_pssm(mat_pth)

    if cleanup:
        os.remove(seq_pth)
        os.remove(msa_pth)
        os.remove(mat_pth)

    return seqid, seq, pssm, aa


def _run_hhblits_multithread(sequences, num_threads=4):

    global pbar

    records = [SeqRecord(Seq(v), k) for k, v in sequences.items()]

    pbar = tqdm(range(len(records)), desc="sequences processed")

    e = ThreadPoolExecutor(num_threads)

    for (seqid, seq, pssm, _) in e.map(_hhblits, records):
        db.pssm.update_one({
            "_id": seqid}, {
            '$set': {"pssm": pssm, "seq": seq}
        }, upsert=True)
        pbar.update(1)


def _run_hhblits(sequences):
    seq_records = [SeqRecord(Seq(seq), id) for id, seq in sequences.items()
                   if not os.path.exists("%s/%s.out" % (out_dir, id))]

    fasta = 'SEQ.fasta'
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


# MUST BE RUN AFTER HHBLITS FINISHED
def _get_pssm(seq):

    # cline = "%s/addss.pl %s.a3m" % (prefix_hhsuite, seq.id)
    # child = subprocess.Popen(cline,
    #                          stdout=subprocess.PIPE,
    #                          stderr=subprocess.PIPE,
    #                          universal_newlines=True,
    #                          shell=(sys.platform != "win32"))
    # handle, _ = child.communicate()
    # assert child.returncode == 0

    cline = "hhfilter -i %s.a3m -o %s.fil.a3m -id 90 -cov 50" % (seq.id, seq.id)
    child = subprocess.Popen(cline,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=(sys.platform != "win32"))

    handle, _ = child.communicate()
    assert child.returncode == 0

    # cline = "%s/reformat.pl -r %s.fil.a3m %s.psi" % (prefix_hhsuite, seq.id, seq.id)
    # child = subprocess.Popen(cline,
    #                          stdout=subprocess.PIPE,
    #                          stderr=subprocess.PIPE,
    #                          universal_newlines=True,
    #                          shell=(sys.platform != "win32"))
    # handle, _ = child.communicate()
    # assert child.returncode == 0

    cline = "%s/reformat.pl -r %s.fil.a3m %s.psi" % (prefix_hhsuite, seq.id, seq.id)
    child = subprocess.Popen(cline,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=(sys.platform != "win32"))
    handle, _ = child.communicate()
    assert child.returncode == 0

    _set_unique_ids("%s.psi" % seq.id, "%s.msa" % seq.id)

    cline = "psiblast -subject %s.seq -in_msa %s.msa -out_ascii_pssm %s.pssm" \
            % (seq.id, seq.id, seq.id)
    child = subprocess.Popen(cline,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=(sys.platform != "win32"))
    _, _ = child.communicate()
    assert child.returncode == 0
    #
    # aln = list(AlignIO.parse(open("%s.fas" % seq.id, 'r'), "fasta"))
    # pssm = SummaryInfo(aln[0]).pos_specific_score_matrix(chars_to_ignore=IGNORE)
    aa, pssm = read_pssm("%s.pssm" % seq.id)

    return seq, pssm


def add_arguments(parser):
    parser.add_argument("--mongo_url", type=str, default='mongodb://localhost:27017/',
                        help="Supply the URL of MongoDB")
    parser.add_argument("--limit", type=int, default=None,
                        help="How many sequences for PSSM computation.")
    parser.add_argument("--max_filter", type=int, default=2000,
                        help="How many sequences to include in the MSA for PSSM computation.")
    parser.add_argument("--num_cpu", type=int, default=2,
                        help="How many cpus for computing PSSM (when running in parallel mode).")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="How many sequences in batch (when running in parallel mode).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    num_cpu = args.num_cpu
    batch_size = args.batch_size
    max_filter = args.max_filter

    prepare_uniprot20()

    client = MongoClient(args.mongo_url)
    db = client['prot2vec']
    lim = args.limit

    seqs = _get_annotated_uniprot(db, lim)
    _run_hhblits_batched(seqs)
