import os
import sys
from tqdm import tqdm

# import A3MIO
from Bio.Seq import Seq
from Bio import SeqIO, AlignIO
from Bio.Align.AlignInfo import SummaryInfo
from Bio.SeqRecord import SeqRecord

from src.python.preprocess import *

from pymongo import MongoClient

from tempfile import gettempdir
tmp_dir = gettempdir()

out_dir = "./hhblits"
if not os.path.exists(out_dir): os.mkdir(out_dir)

prefix_hhsuite = "/usr/share/hhsuite/scripts/"

uniprot20url = "http://wwwuser.gwdg.de/%7Ecompbiol/data/hhsuite/databases/hhsuite_dbs/uniprot20_2016_02.tgz"
uniprot20name = "uniprot20_2016_02"

ASPECT = 'F'
batch_size = 4
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
        os.system("%s/splitfasta.pl %s > /dev/null" % (prefix_hhsuite, sequences_fasta))
        os.system("%s/multithread.pl \'*.seq\' \'hhblits -i $file -d ../dbs/%s/%s -opsi $name.out -n 2 1>out.log 2>err.log\' > /dev/null"
                  % (prefix_hhsuite, uniprot20name, uniprot20name))
        # os.system("%s/multithread.pl \'*.seq\' \'hhblits -i $file -d ../dbs/%s/%s -oa3m $name.a3m -n 2 1>out.log 2>err.log\'"
        #           % (prefix_hhsuite, uniprot20name, uniprot20name))
        os.chdir(pwd)
        pbar.update(j - i)
        i = j

    pbar.close()


# MUST BE RUN AFTER BLITS FINISHED
def _load_pos_specific_score_matrices(sequences):
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


if __name__ == "__main__":
    prepare_uniprot20()

    client = MongoClient("mongodb://127.0.0.1:27017")
    db = client['prot2vec']
    lim = 10

    seqs, _, _ = _get_annotated_uniprot(db, lim)
    _run_parallel_hhblits_msa(seqs)
    _load_pos_specific_score_matrices(seqs)
