__author__ = 'yfrank'

from utils import handleError
from pdb import MotifSelector
from pdb import InterfaceSelector
from pdb import select_structure
from pdb import is_single_domain_protein
from pdb import get_sequences
from pdb import ConnectorPDB
from Bio.PDB import PDBParser

import parameters as params

args = params.arguments
ECOD = args["ecod_fasta"]
PDB = ConnectorPDB(args["pdb_dir"])


class Motif(object):

    def __init__(self, pid, chain, loci=None, seq=None):
        self.pdb = pid
        self.chain = chain
        if not loci:
            self.loci = chain
        else:
            self.loci = loci  # requires a string in ecod format
        self.seq = seq

    @property
    def seq(self):
        return self.__seq

    @seq.setter
    def seq(self,val):
        self.__seq = val

    @property
    def loci(self):
        return self.__loci

    @loci.setter
    def loci(self,val):
        self.__loci = map(Locus, val.split(','))

    @property
    def pdb(self):
        return self.__pdb

    @pdb.setter     # e1htr.1
    def pdb(self, val):
        self.__pdb = val.lower()

    @property
    def chain(self):
        return self.__chain

    @chain.setter
    def chain(self, val):
        self.__chain = val

    @property
    def name(self):
        return "%s%s" % (self.pdb, self.chain)

    def get_pdb_structure(self):
        return PDB.get_structure(self.pdb, parser=PDBParser(QUIET=True))

    def select(self, structure=None):
        if not structure:
            structure = self.get_pdb_structure()
        return select_structure(MotifSelector(self), structure)

    def is_gene_product(self, structure=None):
        if not structure:
            structure = self.get_pdb_structure()
        return is_single_domain_protein(self, structure)

    def get_sequences(self, structure=None):
        if not structure:
            structure = self.get_pdb_structure()
        return get_sequences(structure)

    def __str__(self):
        args = (self.name, self.pdb, self.chain, map(str, self.loci))
        return 'name:\t%s\npdb:\t%s\nchain:\t%s\nloci:\t%s' % args


class Interface(object):
    def __init__(self, motif1, motif2):
        self.from_motif = motif1
        self.to_motif = motif2

    @property
    def from_motif(self):
        return self.__from_motif

    @from_motif.setter
    def from_motif(self, val):
        self.__from_motif = val

    @property
    def to_motif(self):
        return self.__to_motif

    @to_motif.setter
    def to_motif(self, val):
        self.__to_motif = val

    @property
    def from_res(self):
        return self.from_motif.loci

    @from_res.setter
    def from_res(self,val):
        self.from_motif.loci = val

    @property
    def to_res(self):
        return self.to_motif.loci

    @to_res.setter
    def to_res(self,val):
        self.to_motif.loci = val

    @property
    def name(self):
        return "%s_I_%s" % (self.from_motif.name, self.to_motif.name)

    def select(self, structure):
        pdb = self.from_motif.pdb
        assert self.to_motif.pdb == pdb
        return select_structure(InterfaceSelector(self), structure)

    def __str__(self):
        return self.name


class Locus(object):

    def __init__(self, raw):

        self.raw = raw
        if isinstance(raw, str):
            loc = raw.split(':')
            self.chain = loc[0].upper()
            try:
                self.res = loc[1]
            except IndexError as err:
                self.res = "%s-%s" % (int(-10e4), int(10e7))
        else:
            self.chain = raw['chain'].upper()
            self.res = "%s-%s" % (raw['start'], raw['end'])

    @property
    def chain(self):
        return self.__chain

    @chain.setter     # e1htr.1
    def chain(self, val):
        self.__chain = val

    @property
    def res(self):
        return self.__res

    @res.setter     # e1htr.1
    def res(self, val):
        val = val.split('-')
        try:
            val = list(map(int, val))
            l, h = val[0], val[-1]
        except ValueError:     # negative start index
            l, h = -int(val[1]), int(val[2])
        except Exception as err:
            handleError(err, self.raw)
        self.__res = range(l, h+1)

    @property
    def start(self):
        return self.res[0]

    @property
    def end(self):
        return self.res[-1]

    def __str__(self):
        return "%s:%s-%s" % (self.chain, self.res[0], self.res[-1])

    def __repr__(self):
        return {"chain": self.chain, "start": self.start, "end": self.end}


if __name__ == '__main__':
    pass