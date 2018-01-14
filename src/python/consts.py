

exp_codes = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"] + ["TAS", "IC"]


class AminoAcids(object):

    def __init__(self):
        self.aa2index = \
            {
                "A": 0,
                "R": 1,
                "N": 2,
                "D": 3,
                "C": 4,
                "E": 5,
                "Q": 6,
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


AA = AminoAcids()
