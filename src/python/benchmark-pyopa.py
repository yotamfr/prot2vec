import os
import pyopa
import pickle

blast_pth = "../../Data/blast_dist_matrix_04-09-2018"

with open(blast_pth, 'rb') as f:
    dist_mat = pickle.load(f)






pyopa.matrix_dir()


print(os.path.join(pyopa.matrix_dir(), 'logPAM1.json'))
log_pam1_env = pyopa.read_env_json(os.path.join(pyopa.matrix_dir(), 'logPAM1.json'))
s1 = pyopa.Sequence('GCANLVSRLENNSRLLNRDLIAVKINADVYKDPNAGALRL')
s2 = pyopa.Sequence('GCANPSTLETNSQLVNRELIAVKINPRVYKGPNLGAFRL')

# super fast check whether the alignment reaches a given min-score
min_score = 100
pam250_env = pyopa.generate_env(log_pam1_env, 250, min_score)
pyopa.align_short(s1, s2, pam250_env)


