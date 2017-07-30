import os
import gc
import random
import requests
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import distance

from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

from Bio.PDB import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB import Dice

from pymongo import MongoClient

import shutil
import psutil
from utils import CACHE_MANAGER
from utils import ensure_exists

import parameters as params
args = params.arguments
datadir = args["pdb_dir"]
cullpdb = args["cull_pdb"]

client = MongoClient('mongodb://localhost:27017/')
dbname = args["db"]
db = client[dbname]

P = psutil.Process(os.getpid())

PARSE_CIF = os.environ.get('PARSE_CIF')
if not PARSE_CIF: PARSE_CIF = True
print("PARSE_CIF=%s" % PARSE_CIF)
PARSE_CIF = bool(PARSE_CIF)


def list_cull_pdb(limit=None):
    ids = []
    print("Running %s with limit: %s" % (cullpdb, limit))
    with open(cullpdb, 'w+') as f:
        for line in open(cullpdb, 'r'):
            id = line.split(' ')[0][:4]     # leave out the chain data
            ids.append(id.lower())          # switch to lower case
        f.write(','.join(ids[1:]))
    print("found %s domains in file" % (len(ids)-1))
    if not limit: limit = len(ids)
    return list(set(ids[1:limit]))  # leave out the IDs tag


def download_pdb(pid, relpath):
    print("downloading %s->%s" % (pid, relpath))
    req = requests.get('http://files.rcsb.org/download/%s.pdb' % (pid,))
    if req.status_code == 404: # then assume it's a .cif
        req = requests.get('http://files.rcsb.org/download/%s.cif' % (pid,))
    if req.status_code != 200:   # then assume it's a .cif
        raise requests.HTTPError('HTTP Error %s' % req.status_code)
    with open(relpath, 'w+') as f:
        f.write(req.content)


def get_sequences(structure):
    return PPBuilder().build_peptides(structure)


def select_structure(selector, structure):
    new_structure = Structure(structure.id)
    for model in structure:
        if not selector.accept_model(model):
            continue
        new_model = Model(model.id, model.serial_num)
        new_structure.add(new_model)
        for chain in model:
            if not selector.accept_chain(chain):
                continue
            new_chain = Chain(chain.id)
            new_model.add(new_chain)
            for residue in chain:
                if not selector.accept_residue(residue):
                    continue
                new_residue = Residue(residue.id, residue.resname, residue.segid)
                new_chain.add(new_residue)
                for atom in residue:
                    if selector.accept_atom(atom):
                        new_residue.add(atom)
    return new_structure


def is_single_domain_protein(motif, structure):
    selector = MotifSelector(motif)
    locus = motif.loci[0]
    chain_selector = Dice.ChainSelector(locus.chain, int(-10e4), int(10e7))
    protein = select_structure(selector, structure)
    chain = select_structure(chain_selector, structure)
    return len(list(protein.get_atoms())) == len(list(chain.get_atoms()))


def save_pdb(structure, filename, selector=None):
    io = PDBIO()
    io.set_structure(structure)
    if selector:
        io.save(filename, selector) #'1btl-r1.pdb'
    else:
        io.save(filename)


class ConnectorPDB(object):

    def __init__(self, bclean=False):
        self.bclean = bclean
        self.pdb_dir = datadir
        self.cache = CACHE_MANAGER.get_cache('ConnectorPDB.html', expire=60)

    def clean_files(self, pids):
        with open(pids) as f:
            ids = f.read().split(',')
            data_dir = self.pdb_dir
            for item in os.listdir(data_dir):
                path = "%s/%s" % (data_dir, item)
                if not item in ids: shutil.rmtree(path)

    def init_files(self, filename):
        data_dir = self.pdb_dir
        with open(filename) as f:
            ids = f.read().split(',')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for pid in ids:
            pid_dir = "%s/%s" % (data_dir, pid)
            dst = '%s/%s' % (pid_dir, pid)
            old_dst = '%s/%s.pdb' % (pid_dir, pid)
            ensure_exists(pid_dir)
            if os.path.exists(old_dst):
                print("moving %s->%s" % (old_dst, dst))
                shutil.move(old_dst, dst)
            if not os.path.exists(dst) or os.path.getsize(dst) < 100:
                download_pdb(pid, dst)
            else:
                print("found %s\t%10s Bytes" % (dst, os.path.getsize(dst)))

    def get_structure(self, pid, dst=None, parser=None):
        dirname = '%s/%s' % (self.pdb_dir, pid)

        if not dst:
            dst = '%s/%s' % (dirname, pid)
        if not os.path.exists(dst) or os.path.getsize(dst) < 100:
            ensure_exists(dirname)
            download_pdb(pid, dst)

        if not parser:
            parser = PDBParser(QUIET=True)

        def creator(parser=parser):
            try:
                ret = parser.get_structure(pid, file=dst)
            except ValueError as e:  # assume it's a .cif
                if PARSE_CIF:
                    parser = MMCIFParser(QUIET=True)
                    ret = parser.get_structure(pid, dst)
                else:
                    raise e
            finally:
                self.freemem()
            return ret

        if self.cache:  # warn: Leaky Code!
            return self.cache.get(key=pid, createfunc=creator)
        elif os.path.getsize(dst) > consts.PDB_SIZE_LIMIT:
            raise ValueError('file size exceeds %s' % consts.PDB_SIZE_LIMIT)
        else:
            return creator()

    def freemem(self):
        if P.memory_percent() > 45.:
            # print "memory_percent=%8s\tclearing cache..." % (P.memory_percent(),)
            self.cache.clear()
            # gc.collect()

    def select(self, pid, selector, structure=None):
        if not structure:
            structure = self.get_structure(pid)
        structure = select_structure(selector, structure)
        filename = '%s/%s/%s' % (self.pdb_dir, pid, selector.name)
        save_pdb(structure, filename, selector)
        return filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.bclean:
            shutil.rmtree(self.pdb_dir)

    #def __str__(self):
     #   return ' '.join(self.structs.keys())


class InterfaceSelector(Dice.ChainSelector):

    def __init__(self, interface):

        Selector = Dice.ChainSelector
        loci = interface.from_motif.loci + interface.to_motif.loci
        self.selectors = [Selector(locus.chain, locus.start, locus.end) for locus in loci]
        self.interface = interface
        self.num_atoms = 0
        self.num_res = 0

    def accept_model(self, model):
        for selector in self.selectors:
            if selector.accept_model(model):
                return 1
        return 0

    def accept_chain(self, chain):
        for selector in self.selectors:
            if selector.accept_chain(chain):
                return 1
        return 0

    def accept_residue(self, residue):
        hetatm_flag, resseq, icode = residue.get_id()
        residue.id = (hetatm_flag, resseq, ' ')     # evade selection errors
        for selector in self.selectors:
            if selector.accept_residue(residue):
                self.num_res += 1
                return 1
        return 0

    def accept_atom(self, atom):
        for selector in self.selectors:
            if selector.accept_atom(atom):
                self.num_atoms += 1
                return 1
        return 0

    @property
    def name(self):
        return self.interface.name


class MotifSelector(Dice.ChainSelector):

    def __init__(self, motif):
        Selector = Dice.ChainSelector
        self.selectors = [Selector(locus.chain, locus.start, locus.end) for locus in motif.loci]
        self.motif = motif
        self.num_atoms = 0
        self.num_res = 0

    def accept_model(self, model):
        for selector in self.selectors:
            if selector.accept_model(model):
                return 1
        return 0

    def accept_chain(self, chain):
        for selector in self.selectors:
            if selector.accept_chain(chain):
                return 1
        return 0

    def accept_residue(self, residue):
        hetatm_flag, resseq, icode = residue.get_id()
        residue.id = (hetatm_flag, resseq, ' ')     # evade selection errors
        for selector in self.selectors:
            if selector.accept_residue(residue):
                self.num_res += 1
                return 1
        return 0

    def accept_atom(self, atom):
        for selector in self.selectors:
            if selector.accept_atom(atom):
                self.num_atoms += 1
                return 1
        return 0

    @property
    def name(self):
        return self.motif.name


def closestnbr(A, B):
    kdt = KDTree(A)
    nearest = map(lambda b: kdt.query(b), B)
    return min(map(lambda p: p[0], nearest))


def get_distance_matrix(atoms1, atoms2, metric=distance.euclidean):
    A = np.matrix(map(lambda a: a.get_coord(), atoms1))
    B = np.matrix(map(lambda a: a.get_coord(), atoms2))
    return distance.cdist(A, B, metric)


def measure_distances(model1, model2):
    atoms1 = list(model1.get_atoms())
    atoms2 = list(model2.get_atoms())
    calphas1 = filter(lambda a: a.name=='CA', atoms1)
    calphas2 = filter(lambda a: a.name=='CA', atoms2)
    atomMatom = get_distance_matrix(atoms1, atoms2)
    caMca = get_distance_matrix(calphas1, calphas2)
    return [atomMatom.min(), caMca.min(),
           atomMatom.max(), caMca.max(),
           atomMatom.mean(), caMca.mean()]


def measure_min_dist(model1, model2):
    def get_min_dist(atoms1, atoms2):
        A = np.matrix(map(lambda a: a.get_coord(), atoms1))
        B = np.matrix(map(lambda a: a.get_coord(), atoms2))
        return closestnbr(A, B)
    atoms1 = list(model1.get_atoms())
    atoms2 = list(model2.get_atoms())
    cas1 = filter(lambda a: a.name == 'CA', atoms1)
    cas2 = filter(lambda a: a.name == 'CA', atoms2)
    atomMatom = get_min_dist(atoms1, atoms2)
    caMca = get_min_dist(cas1, cas2)
    return [atomMatom[0], caMca[0]]


if __name__ == '__main__':
    print(closestnbr([(0, 0), (7, 6), (2, 20), (12, 5), (16, 16)],
                     [(5, 8), (19, 7), (14, 22), (8, 19), (7, 29), (10, 11), (1, 13)]))
    print(distance.euclidean((7, 6), (5, 8)))