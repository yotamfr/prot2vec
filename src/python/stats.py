from fastsemsim.Ontology import ontologies
from fastsemsim.SemSim.CosineSemSim import CosineSemSim
from fastsemsim.SemSim.GSESAMESemSim import GSESAMESemSim
from fastsemsim.SemSim.BMASemSim import BMASemSim
from fastsemsim.Ontology import AnnotationCorpus
from fastsemsim.SemSim.SemSimUtils import SemSimUtils
from fastsemsim import data

from pymongo import MongoClient
import pandas as pd

from itertools import combinations
from scipy.misc import comb

from alignment import sequence_identity_by_id

from gensim.models.word2vec import Word2Vec

import sys

from models import EcodDomain
from models import PdbChain

import parameters as params
import utils
from prot2vec import Node2Vec

args = params.arguments
logger = utils.get_logger("results")

ckptpath = args["ckpt_path"]
seq_length = args["seq_length"]
emb_dim = args["word_embedding_dim"]
datapath = args["data_path"]

client = MongoClient('mongodb://localhost:27017/')
dbname = args["db"]
db = client[dbname]


def load_goa():
    # Select the type of ontology (GeneOntology, ...)
    ontology_type = 'GeneOntology'
    # ontology_type = 'CellOntology'
    # ontology_type = 'DiseaseOntology'

    # Select the relatioships to be ignored. For the GeneOntology, has_part is ignore by default, for CellOntology, lacks_plasma_membrane_part is ignored by default
    # ontology_parameters =	{}
    ontology_parameters = {'ignore': {}}
    # ontology_parameters =	{'ignore':{'has_part':True, 'occurs_in':True, 'happens_during':True}}
    # ontology_parameters =	{'ignore':{'regulates':True, 'has_part':True, 'negatively_regulates':True, 'positively_regulates':True, 'occurs_in':True, 'happens_during':True}}

    # Select the source file type (obo or obo-xml)
    ontology_source_type = 'obo'

    # Select the ontology source file name. If None, the default ontology_type included in fastsemsim will be used
    ontology_source = None

    # Select the ac source file name. If None, the default ac included in fastsemsim for the ac_species will be used
    ac_source = None

    ac_species = 'human'
    # ac_species = 'arabidopsis'
    # ac_species = 'fly'
    # ac_species = 'mouse'
    # ac_species = 'rat'
    # ac_species = 'worm'
    # ac_species = 'zebrafish'

    ac_source_type = 'plain'
    # ac_source_type = 'gaf-2.0'

    ac_params = {}

    # gaf-2.0 ac
    ac_params['filter'] = {}  # filter section is useful to remove undesired annotations
    ac_params['filter']['EC'] = {}  # EC filtering: select annotations depending on their EC
    # ac_params['filter']['EC']['EC'] = EC_include # select which EC accept or reject
    # ac_params['filter']['EC']['inclusive'] = True # select which EC accept or reject
    # ac_params['filter']['EC'] = {} # EC filtering: select annotations depending on their EC
    # ac_params['filter']['EC']['EC'] = EC_ignore # select which EC accept or reject
    # ac_params['filter']['EC']['inclusive'] = False # select which EC accept or reject

    ac_params['filter']['taxonomy'] = {}
    # ac_params['filter']['taxonomy']['taxonomy'] = tax_include # set properly this field to load only annotations involving proteins/genes of a specific species
    # ac_params['filter']['taxonomy']['inclusive'] = True # select which EC accept or reject
    # ac_params['filter']['taxonomy'] = {}
    # ac_params['filter']['taxonomy']['taxonomy'] = tax_ignore
    # ac_params['filter']['taxonomy']['inclusive'] = False # select which EC accept or reject
    # ac_params['simplify'] = True # after parsing and filtering, removes additional information such as taxonomy or EC. Useful if you have a huge amount of annotations and not enough memory

    # Plain ac
    ac_params[
        'multiple'] = True  # Set to True if there are many associations per line (the object in the first field is associated to all the objects in the other fields within the same line)
    ac_params[
        'term first'] = False  # set to True if the first field of each row is a GO term. Set to False if the first field represents a protein/gene
    ac_params['separator'] = "\t"  # select the separtor used to divide fields

    print("\n#######################")
    print("# Loading ontology... #")
    print("#######################\n")

    ontology = ontologies.load(source=ontology_source, source_type=ontology_source_type, ontology_type=ontology_type,
                               parameters=ontology_parameters)

    print("\n######################")
    print("# Loading annotation corpus... #")
    print("######################\n")

    ac = AnnotationCorpus.AnnotationCorpus(ontology)
    builtin_dataset = data.dataset.Dataset()
    if ac_source is None:
        ac_data = builtin_dataset.get_default_annotation_corpus(ontology.name, ac_species)
        ac_source = ac_data['file']
        ac_source_type = ac_data['filetype']
    if ac_source is None:
        ac = None
    ac.parse(ac_source, ac_source_type, ac_params)
    ac.isConsistent()

    print("\n#################################")
    print("# Annotation corpus successfully loaded.")
    print("#################################\n")

    return ontology, ac


def load_geneontology():
    # Select the type of ontology (GeneOntology, ...)
    ontology_type = 'GeneOntology'
    # ontology_type = 'CellOntology'
    # ontology_type = 'DiseaseOntology'

    # Select the relatioships to be ignored. For the GeneOntology, has_part is ignore by default, for CellOntology, lacks_plasma_membrane_part is ignored by default
    # ignore_parameters =	{}
    ignore_parameters = {'ignore': {}}
    # ignore_parameters =	{'ignore':{'has_part':True, 'occurs_in':True, 'happens_during':True}}
    # ignore_parameters =	{'ignore':{'regulates':True, 'has_part':True, 'negatively_regulates':True, 'positively_regulates':True, 'occurs_in':True, 'happens_during':True}}

    # Select the source file type (obo or obo-xml)
    source_type = 'obo'

    # Select the source file name. If None, the default GeneOntology included in fastsemsim will be used
    # source = None
    source = args["obo_file"]

    print("\n#######################")
    print("# Loading ontology... #")
    print("#######################\n")

    ontology = ontologies.load(source=source, source_type=source_type, ontology_type=ontology_type,
                               parameters=ignore_parameters)

    print("\n#################################")
    print("# Ontology successfully loaded. #")
    print("#################################\n")

    return ontology


class Term(object):

    def __init__(self, term):
        detail = term.findall('detail')[0]
        self.goid = term.get('id')
        self.pid = term.get('structureId')
        self.chain = term.get('chainId')
        self.name = detail.get('name')
        self.definition = detail.get('definition')
        self.ontology = detail.get('ontology')

    def __str__(self):
        return '%s\t%s' % (self.goid, self.name)

# GO, AC = load_goa()
GO = load_geneontology()
Utils = SemSimUtils(GO, ac=None)
TSS_GSESAME = GSESAMESemSim(GO, util=Utils)
TSS_Cosine = CosineSemSim(GO, util=Utils)
MTSS_BMA = BMASemSim(GO, ac=None, util=Utils)


class StatisticsBar(object):
    def __init__(self, n):
        self.n = n
        self.indx = 0
        self.right = 0
        self.wrong = 0
        self.sum_delta = 0
        sys.stdout.write('\n')

    def display(self):
        percright = 100.0*self.right/self.indx
        percwrong = 100.0*self.wrong/self.indx
        perccheck = 100.0*self.indx/self.n
        avgdelta = self.sum_delta / self.indx
        sys.stdout.write(
            "\rGot {0}({4:.2f}%) Right vs {1}({5:.2f}%) Wrong, Checked {2}({7:.2f}%) out of {3} p.. Avg. delta is {6}"
                .format(self.right, self.wrong, self.indx, self.n, percright, percwrong, avgdelta, perccheck))

    def update(self, delta, inc):
        self.indx += inc
        self.sum_delta += delta
        if delta > .5:
            self.incwrong()
        if delta < .25:
            self.incright()

    def incright(self, i=1):
        self.right += i
        self.display()

    def incwrong(self, i=1):
        self.wrong += i
        self.display()

def compute_semsim(terms1, terms2, func=MTSS_BMA.SemSim, metric=TSS_Cosine):
    return func(list(map(lambda s: s.encode("ascii", "ignore"), terms1)),
                list(map(lambda s: s.encode("ascii", "ignore"), terms2)), TSS=metric)


def compute_similarity(model, annots):
    ids = annots.keys()
    pairs = combinations(ids, 2)
    STATS, n = [], int(comb(len(ids), 2))
    logger.info("Searching %s pairs" % n)
    bar = StatisticsBar(n)
    for pair in pairs:
        try:
            semsim1 = model.similarity(pair[0], pair[1])
            goids1, goids2 = annots[pair[0]], annots[pair[1]]
            semsim2 = compute_semsim(goids1, goids2)
            #seqid = sequence_identity_by_id(pair[0], pair[1], db.ecod)
            seqid = -1
            record = [pair[0], pair[1], semsim1, semsim2, seqid]
            delta = abs(semsim1 - semsim2)
            STATS.append(record)
            bar.update(delta, 1)
            if semsim2 > .9 and delta < .25:
                #logger.debug(record)
                pass
        except (TypeError, KeyError) as err:
            pass
    return STATS


def save_stats(stats, filename, cols):
    print ('\nSaving %s Records to %s' % (len(stats), filename))
    df = pd.DataFrame(stats, columns=cols)
    df.to_csv(filename, index=False, encoding='utf-8')
    return df


def benchmark_pdb_embedding(method, sample_size):
    logger.info("Benchmarking Results...")
    if method == 'prot2vec':
        annots = {p['_id']: list(map(lambda a: a['GO_ID'],
                                     db.goa.find({'DB_Object_ID': p['_id']})))
                  for p in db.pdb.aggregate([{"$sample": {"size": sample_size}}])}
        model = Word2Vec.load('%s/%s.%s.model' % (ckptpath, method, emb_dim))
        output = '%s/%s.%s.semsim.csv' % (ckptpath, method, emb_dim)
    elif method == 'ecod.dense':
        annots = {dom.eid: dom.get_go_terms() for dom in
                  filter(lambda e: e.get_go_terms(),
                         map(EcodDomain, db.ecod.aggregate([{"$sample": {"size": sample_size}}])))}
        model = Node2Vec()
        model.load("%s/ecod.dense.emb" % ckptpath)
        output = '%s/ecod.dense.semsim.csv' % ckptpath
    elif method == 'ecod.combined':
        annots = {dom.eid: dom.get_go_terms() for dom in
                  filter(lambda e: e.get_go_terms(),
                         map(EcodDomain, db.ecod.aggregate([{"$sample": {"size": sample_size}}])))}
        model = Node2Vec()
        model.load("%s/ecod.dense.emb" % ckptpath)
        model.load("%s/ecod.simple.emb" % ckptpath)
        output = '%s/ecod.combined.semsim.csv' % ckptpath
    elif method == 'pdb.95':
        annots = {chain.pid: chain.get_go_terms() for chain in
                  filter(lambda p: p.get_go_terms(),
                         map(PdbChain, db.pdb.aggregate([{"$sample": {"size": sample_size}}])))}
        model = Node2Vec()
        model.load("%s/pdb.95.emb" % ckptpath)
        output = '%s/pdb.95.semsim.csv' % ckptpath
    elif method == 'pdb.60':
        annots = {chain.pid: chain.get_go_terms() for chain in
                  filter(lambda p: p.get_go_terms(),
                         map(PdbChain, db.pdb.aggregate([{"$sample": {"size": sample_size}}])))}
        model = Node2Vec()
        model.load("%s/pdb.60.emb" % ckptpath)
        output = '%s/pdb.60.semsim.csv' % ckptpath
    else:
        logger.error("Unknown method")
        return

    save_stats(compute_similarity(model, annots), output,
               cols=['PROT1', 'PROT2', 'PROT2VEC', 'GENEONTOLOGY', 'SEQ_IDENTITY'])


if __name__ == '__main__':
    print MTSS_BMA.SemSim(['GO:0000786', 'GO:0005634', 'GO:0003677', 'GO:0007283'],
                          ['GO:0004672', 'GO:0005524', 'GO:0006468'], TSS=TSS_Cosine)

    # benchmark_pdb_embedding("prot2vec", 1000)
    benchmark_pdb_embedding("pdb.60", 10**4)
