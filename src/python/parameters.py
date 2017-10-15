"""
The hyperparameters for a model are defined here. Arguments like the type of model, model name, paths to data, logs etc. are also defined here.
All paramters and arguments can be changed by calling flags in the command line.

Required arguements are,
model_type: which model you wish to train with. Valid model types: cbow, bilstm, and esim.
model_name: the name assigned to the model being trained, this will prefix the name of the logs and checkpoint files.
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--datapath", type=str, default="../../data")
parser.add_argument("--ckptpath", type=str, default="../../models")

parser.add_argument("--db", metavar='database name', type=str, choices=['cafa3', 'prot2vec'],
                    dest="dbname", default="prot2vec", help="Give DB name.")

parser.add_argument("--learning_rate", type=float, default=0.0004, help="Learning rate for model")
parser.add_argument("--seq_length", type=int, default=50, help="Max sequence length")
parser.add_argument("--max_paths", type=int, default=10**7, help="Max paths to sample")

args = parser.parse_args()

arguments = {

    "uniprot_sprot_fasta": "{}/Uniprot/uniprot_sprot.fasta".format(args.datapath),
    "uniprot_trembl_fasta": "{}/Uniprot/uniprot_trembl.fasta".format(args.datapath),

    "goa_uniprot_all": "{}/GO/goa_uniprot_all.gpa".format(args.datapath),

    "cafa3_sprot_goa": "{}/cafa/CAFA3_training_data/uniprot_sprot_exp.txt".format(args.datapath),
    "cafa3_sprot_seq": "{}/cafa/CAFA3_training_data/uniprot_sprot_exp.fasta".format(args.datapath),

    "ckpt_path": "{}".format(args.ckptpath),
    "data_path": "{}".format(args.datapath),

    "db": args.dbname,
    "word_embedding_dim": 128,
    "seq_length": args.seq_length,
    "batch_size": 32,
    "learning_rate": args.learning_rate,

    "pdb_fasta": "{}/pdbaa".format(args.datapath),
    "pdbnr_fasta": "{}/pdbaa.nr".format(args.datapath),
    "ecod_fasta": "{}/ecod.latest.fasta.txt".format(args.datapath),
    # "pdb_gaf": "{}/goa_pdb.gaf".format(args.datapath),
    "cull_pdb": "{}/PDB/cull_pdb.txt".format(args.datapath),
    "pdb_dir": "{}/PDB/structures".format(args.datapath),

    # "obo_file": "{}/go-basic.obo".format(args.datapath),

    # "biogrid_organism": "{}/BioGRID/BIOGRID-ORGANISM-3.4.152.tab2".format(args.datapath),
    # "biogrid_all": "{}/BioGRID/BIOGRID-ALL-3.4.152.tab2.txt".format(args.datapath),
    # "biogrid_ids": "{}/BioGRID/BIOGRID-IDENTIFIERS-3.4.152.tab.txt".format(args.datapath),
    #
    # "intact_all": "{}/IntAct/all/psimitab/intact.txt".format(args.datapath),
    #
    # "clstr_pdb": "{}/pdb.95.clstr".format(args.datapath),
    # "clstr_ecod": "{}/ecod.95.clstr".format(args.datapath),

}
