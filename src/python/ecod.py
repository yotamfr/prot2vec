__author__ = 'yfrank'

from utils import ispdbid
from utils import isecodid
from utils import handleError
from domain import Locus
from pdb import MotifSelector
from pdb import ConnectorPDB
from pdb import select_structure
from pdb import is_single_domain_protein
from pymongo import MongoClient

import parameters as params
args = params.arguments
cullpdb = args["cull_pdb"]
ECOD = args['ecod_fasta']

client = MongoClient('mongodb://localhost:27017/')
dbname = args["db"]
db = client[dbname]


class IllegalArgumentError(ValueError):
    pass


class EcodDomain(object):

    def __init__(self, *args, **kwargs):

        header = kwargs.get('header')

        if not header:
            d = args[0]
            self.uid = d['uid']
            self.eid = d['ecod_id']
            self.hierarchy = d['hierarchy']
            self.loci = d['loci']
            self.seq = d['sequence']
            self.clstr = -1 if 'clstr' not in d else d['clstr']
        else:  # header=e1htr.1|1.1.1.1|P:1-43,B:1-329
            args = header.strip().split('|')
            self.uid = args[0]
            self.eid = args[1]
            self.hierarchy = args[2]
            self.loci = args[3]
            self.seq = kwargs.get('sequence')

    @property
    def seq(self):
        return self.__seq.strip()

    @seq.setter
    def seq(self,val):
        self.__seq = val

    @property
    def loci(self):
        return self.__loci

    @loci.setter
    def loci(self, val):
        try:
            self.__loci = map(Locus, val.split(','))
        except AttributeError:
            self.__loci = map(Locus, val)

    @property
    def eid(self):
        return self.__eid

    @property
    def name(self):
        return self.__eid

    @eid.setter     # e1htr.1
    def eid(self, val):
        self.__eid = val

    @property
    def chain(self):
        suffix = self.__eid[5:]
        return suffix[:-1]

    @property
    def num(self):
        suffix = self.__eid[5:]
        return int(suffix[-1])

    @property
    def pdb(self):
        return self.__eid[1:5].upper()

    @property
    def complex(self):
        return self.__eid[1:5].upper()

    def is_gene_product(self, structure=None):
        if not structure:
            record = db.ecod.find_one({})
        else:
            return is_single_domain_protein(self, structure)

    def select(self, structure):
        return select_structure(MotifSelector(self), structure)

    def get_adj_nodes(self):
        query =[]
        query.extend([{"complex": self.complex, "loci.chain": locus.chain, "loci.start": locus.end+1}
                      for locus in self.loci])
        query.extend([{"complex": self.complex, "loci.chain": locus.chain, "loci.end": locus.start-1}
                      for locus in self.loci])
        query.extend([{"complex": self.complex, "chain": locus.chain, "num": self.num+1}
                      for locus in self.loci])
        query.extend([{"complex": self.complex, "chain": locus.chain, "num": self.num-1}
                      for locus in self.loci])
        return map(EcodDomain, db.ecod.find({'$or': query}))

    def get_go_terms(self):
        return reduce(lambda x, y: x | y,
                      [set(map(lambda e: e["GO_ID"], db.goa.find({"PDB_ID": self.pdb, "Chain": locus.chain})))
                       for locus in self.loci], set())

    def __str__(self):
        args = (self.eid, self.pdb, self.chain, self.num, map(str, self.loci), self.hierarchy)
        return 'eid:\t%s\npdb:\t%s\nchain.#:\t%s.%s\nloci:\t%s\nhierarchy:%s' % args


class Extractor(object):

    def __init__(self, src=ECOD):
        self.ecod = open(src, 'r')
        self.lines = self.ecod.read()
        self.ecod.seek(0)

    # return position or -1 if domain doesn't exist
    def contains(self, domain, start=0):
        if isecodid(domain):
            answer = self.lines.find(domain, start)
            return answer
        else:
            raise IllegalArgumentError("%s is not an ecod full domain id" % domain)

    def extract_domains(self, pdb):
        assert not self.ecod.tell()
        domains = []
        pid = pdb.lower()
        if not ispdbid(pid):
            raise IllegalArgumentError("value is not an ecod pdb id")
        pos = self.lines.find(pid, 0)
        while pos != -1:
            self.ecod.seek(pos)
            h = self.ecod.next()
            s = self.ecod.next()
            domains.append(EcodDomain(header=h, sequence=s))
            pos = self.lines.find(pid, self.ecod.tell())
        self.ecod.seek(0)
        return domains

    def extract_all_domains(self):
        assert not self.ecod.tell()
        pdbs = {}
        while True:
            try:
                h = self.ecod.next()
                s = self.ecod.next()
            except StopIteration:
                break
            h = h[h.find('e'):]
            domain = EcodDomain(header=h, sequence=s)
            key = domain.pdb
            if key in pdbs:
                pdbs[key].append(domain)
            else:
                pdbs[key] = [domain]
        self.ecod.seek(0)
        return pdbs

    def extract(self, eid):
        assert not self.ecod.tell()
        try:
            pos = self.contains(eid)
            if pos != -1:
                self.ecod.seek(pos+1)   # offset due to 'e'
                h = self.ecod.next()
                s = self.ecod.next()
                return {'head': h, 'seq': s}
        except IllegalArgumentError as err:
            handleError(err)
        finally:
            self.ecod.seek(0)

    def extract_interaction(self):
        pass

    def __del__(self):
        self.ecod.close()


def get_ecod_domain(id, extractor=None):
    return EcodDomain(db.ecod.find_one({"ecod_id": id}))


def main():
    with ConnectorPDB() as PDB:
        dom1 = get_ecod_domain(id='e1htr.1')
        dom2 = get_ecod_domain(id='e2bjuA2')
        print(dom1.get_adj_nodes())
        print(dom2.get_adj_nodes())

        print(dom1.get_go_terms())
        print(dom2.get_go_terms())

        # PDB.init_files(cullpdb)
        model1 = PDB.get_structure('2w7e')
        model2 = PDB.get_structure('4y4o')

        selection1 = dom1.select(model1)
        selection2 = dom2.select(model2)

        print(selection1)
        print(selection2)


if __name__ == '__main__':
    main()
