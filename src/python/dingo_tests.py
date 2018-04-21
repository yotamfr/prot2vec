import torch

from src.python.preprocess2 import *

from src.python.geneontology import *

from src.python.baselines import *

from src.python.dingo_utils import *

from src.python.dingo_net2 import *

from tqdm import tqdm

import itertools



def kolmogorov_smirnov_cosine(pos, neg, metric):
    data1 = run_metric_on_triplets(metric, pos)
    data2 = run_metric_on_triplets(metric, neg)
    save_object(data1, "Data/dingo_%s_ks_cosine_pos_data" % asp)
    save_object(data2, "Data/dingo_%s_ks_cosine_neg_data" % asp)
    return ks_2samp(data1, data2)


def kolmogorov_smirnov_norm(pos, neg, metric):
    data1 = run_metric_on_pairs(metric, pos)
    data2 = run_metric_on_pairs(metric, neg)
    save_object(data1, "Data/dingo_%s_ks_norm_pos_data" % asp)
    save_object(data2, "Data/dingo_%s_ks_norm_neg_data" % asp)
    return ks_2samp(data1, data2)


if __name__ == "__main__":

    cleanup()

    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['prot2vec']

    asp = 'F'   # molecular function
    onto = get_ontology(asp)
    t0 = datetime.datetime(2014, 1, 1, 0, 0)
    t1 = datetime.datetime(2014, 9, 1, 0, 0)
    # t0 = datetime.datetime(2017, 1, 1, 0, 0)
    # t1 = datetime.datetime.utcnow()

    print("Indexing Data...")
    trn_stream, tst_stream = get_training_and_validation_streams(db, t0, t1, asp)
    print("Loading Training Data...")
    uid2seq_trn, _, go2ids_trn = trn_stream.to_dictionaries(propagate=True)
    print("Loading Validation Data...")
    uid2seq_tst, _, go2ids_tst = tst_stream.to_dictionaries(propagate=True)

    print("Building Graph...")
    graph = Graph(onto, uid2seq_trn, go2ids_trn)
    print("Graph contains %d nodes" % len(graph))

    print("Load DigoNet")
    go_embedding_weights = np.asarray([onto.todense(go) for go in onto.classes])
    net = AttnDecoder(ATTN, 100, 10, go_embedding_weights)
    net = net.cuda()
    # ckpth = "/tmp/digo_0.01438.tar"
    ckpth = "/tmp/digo_0.15157.tar"
    print("=> loading checkpoint '%s'" % ckpth)
    checkpoint = torch.load(ckpth, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])

    print("Running K-S tests...")
    pos, neg = sample_pos_neg(graph)
    data_pos, data_neg = [], []
    for (p_s1, p_s2, p_n), (n_s1, n_s2, n_n) in zip(pos, neg):
        data_pos.append((p_s1, p_n))
        data_pos.append((p_s2, p_n))
        data_neg.append((n_s1, n_n))
        data_neg.append((n_s2, n_n))
    compute_vectors(data_pos, net, onto)
    compute_vectors(data_neg, net, onto)
    res = kolmogorov_smirnov_norm(data_pos, data_neg, l2_norm)
    print("K-S l2_norm: %s, %s" % res)
    res = kolmogorov_smirnov_cosine(pos, neg, cosine_similarity)
    print("K-S cosine: %s, %s" % res)