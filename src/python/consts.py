from datetime import datetime
from Bio.SubsMat import MatrixInfo

exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"] + ["TAS", "IC"] + ["HDA", "HEP", "HMP"]

t0 = datetime(2014, 1, 1, 0, 0)
t1 = datetime(2014, 9, 1, 0, 0)

cafa2_cutoff = datetime(2014, 1, 1, 0, 0)
cafa3_cutoff = datetime(2017, 2, 2, 0, 0)

TODAY = today_cutoff = datetime.now()

NOW = datetime.utcnow()

PAD = 25


class AminoAcids(object):

    def __len__(self):
        return 20

    def __init__(self):
        self.aa2index = \
            {
                "A": 0,
                "R": 1,
                "N": 2,
                "D": 3,
                "C": 4,
                "Q": 5,
                "E": 6,
                "G": 7,
                "H": 8,
                "I": 9,
                "L": 10,
                "K": 11,
                "M": 12,
                "F": 13,
                "P": 14,
                "S": 15,
                "T": 16,
                "W": 17,
                "Y": 18,
                "V": 19,
                "X": 20,
                "B": 21,
                "Z": 22,
                "O": 23,
                "U": 24
            }
        self.index2aa = {v: k for k, v in self.aa2index.items()}

        self.aa2onehot = {

            "A": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "R": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "N": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "D": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "C": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Q": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "E": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "G": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "H": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "I": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "L": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "K": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "M": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "F": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "P": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "S": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "T": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "W": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "V": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "X": [.05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05],
            "B": [0, 0, .5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # D, N
            "Z": [0, 0, 0, 0, 0, .5, .5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # E, Q
            "O": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],      # O -> K
            "U": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]       # U -> C

        }

        self.blosum62 = {k: [0] * 25 for k in self.aa2index.keys()}
        for k1, v1 in self.aa2index.items():
            if k1 == "O" or k1 == "U":
                continue
            for k2, v2 in self.aa2index.items():
                if k2 == "O": k2 = "K"
                elif k2 == "U": k2 = "C"
                self.blosum62[k1][v2] = MatrixInfo.blosum62[(k1, k2)] \
                    if (k1, k2) in MatrixInfo.blosum62 else MatrixInfo.blosum62[(k2, k1)]

        self.blosum62['O'][:] = self.blosum62['K'][:]
        self.blosum62['U'][:] = self.blosum62['C'][:]


AA = AminoAcids()


if __name__=="__main__":
    print(AA.blosum62)
