
from tqdm import tqdm

import numpy as np

import itertools


def sample_pairs(nodes, include_nodes=False, sample_size=10000):
    pairs = set()
    pbar = tqdm(range(len(nodes)), desc="nodes sampled")
    for node in nodes:
        pbar.update(1)
        s_in = min(200, node.size)
        sample_in = np.random.choice(list(node.sequences), s_in, replace=False)
        if include_nodes:
            pairs |= set((seq1, seq2, node, node) for seq1, seq2 in itertools.combinations(sample_in, 2))
        else:
            pairs |= set((seq1, seq2) for seq1, seq2 in itertools.combinations(sample_in, 2))
    pbar.close()
    n = len(pairs)
    pairs_indices = np.random.choice(list(range(n)), min(n, sample_size), replace=False)
    return np.asarray(list(pairs))[pairs_indices, :]


def sample_pairs_iou(graph, include_nodes=False, sample_size=10000):
    data = set()
    leaf_pairs = list(itertools.combinations(list(graph.leaves), 2))
    n = len(leaf_pairs)
    indices = np.random.choice(list(range(n)), sample_size, replace=False)
    pbar = tqdm(range(len(indices)), desc="nodes sampled")
    for leaf1, leaf2 in np.asarray(leaf_pairs)[indices, :]:
        intersection = leaf1.ancestors & leaf2.ancestors
        union = leaf1.ancestors | leaf2.ancestors
        iou = len(intersection) / len(union)
        iou = 2 * iou - 1                       # scale to [-1, 1]
        sequences1 = list(leaf1.sequences - leaf2.sequences)
        sequences2 = list(leaf2.sequences - leaf1.sequences)
        s1 = min(len(sequences1), 100)
        sample1 = np.random.choice(list(sequences1), s1, replace=False) if sequences1 else []
        s2 = min(len(sequences2), 100)
        sample2 = np.random.choice(list(sequences2), s2, replace=False) if sequences2 else []
        if include_nodes:
            data |= set((seq1, seq2, leaf1, leaf1, 1) for seq1, seq2 in itertools.combinations(sample1, 2))
            data |= set((seq1, seq2, leaf2, leaf2, 1) for seq1, seq2 in itertools.combinations(sample2, 2))
            data |= set((seq1, seq2, leaf1, leaf2, iou) for seq1 in sample1 for seq2 in sample2)
            data |= set((seq2, seq1, leaf2, leaf1, iou) for seq2 in sample2 for seq1 in sample1)
        else:
            data |= set((seq1, seq2, 1) for seq1, seq2 in itertools.combinations(sample1, 2))
            data |= set((seq1, seq2, 1) for seq1, seq2 in itertools.combinations(sample2, 2))
            data |= set((seq1, seq2, iou) for seq1 in sample1 for seq2 in sample2)
            data |= set((seq2, seq1, iou) for seq2 in sample2 for seq1 in sample1)
        pbar.update(1)
    pbar.close()
    n = len(data)
    indices = np.random.choice(list(range(n)), min(n, sample_size), replace=False)
    return np.asarray(list(data))[indices, :]


def sample_pos_neg(graph, include_nodes=False, sample_size=10000):
    pos, neg = set(), set()
    pbar = tqdm(range(len(graph)), desc="nodes sampled")
    for node in graph:
        pbar.update(1)
        if not node.is_leaf():
            continue
        s_in = min(100, node.size)
        sample_in = np.random.choice(list(node.sequences), s_in, replace=False)
        if include_nodes:
            pos |= set((seq1, seq2, node, node) for seq1, seq2 in itertools.combinations(sample_in, 2))
        else:
            pos |= set((seq1, seq2) for seq1, seq2 in itertools.combinations(sample_in, 2))
        for cousin in node.cousins:
            cousin_sequences = cousin.sequences - node.sequences
            if not cousin_sequences:
                continue
            s_out = min(100, len(cousin_sequences))
            sample_out = np.random.choice(list(cousin_sequences), s_out, replace=False)
            if include_nodes:
                neg |= set((seq1, seq2, node, cousin) for seq1 in sample_in for seq2 in sample_out)
            else:
                neg |= set((seq1, seq2) for seq1 in sample_in for seq2 in sample_out)
    pbar.close()
    n, m = len(pos), len(neg)
    pos_indices = np.random.choice(list(range(n)), min(n, sample_size), replace=False)
    neg_indices = np.random.choice(list(range(m)), min(m, sample_size), replace=False)
    return np.asarray(list(pos))[pos_indices, :], np.asarray(list(neg))[neg_indices, :]


def sample_pos_neg_no_common_ancestors(graph, include_nodes=False, sample_size=10000):
    pos, neg = set(), set()
    root_children = set(graph.root.children)
    seq2nodes = {}
    for node in graph:
        for seq in node.sequences:
            if seq in seq2nodes:
                seq2nodes[seq].add(node)
            else:
                seq2nodes[seq] = {node}
    pbar = tqdm(range(len(graph)), desc="nodes sampled")
    for node in graph:
        pbar.update(1)
        if not node.is_leaf():
            continue
        list_in = list(node.sequences)
        s_in = min(100, len(list_in))
        sample_in = np.random.choice(list_in, s_in, replace=False)
        if include_nodes:
            pos |= set((seq1, seq2, node, node) for seq1, seq2 in itertools.combinations(sample_in, 2))
        else:
            pos |= set((seq1, seq2) for seq1, seq2 in itertools.combinations(sample_in, 2))
        non_ancestors = root_children - node.ancestors
        if not non_ancestors:
            continue
        distant = np.random.choice(list(non_ancestors))
        for child in distant.descendants:
            if not child.is_leaf():
                continue
            list_out = list(filter(lambda s: node not in seq2nodes[s], child.sequences))
            if not list_out:
                continue
            s_out = min(100, len(list_out))
            sample_out = np.random.choice(list_out, s_out, replace=False)
            if include_nodes:
                neg |= set((seq1, seq2, node, child) for seq1 in sample_in for seq2 in sample_out)
            else:
                neg |= set((seq1, seq2) for seq1 in sample_in for seq2 in sample_out)
    pbar.close()
    n, m = len(pos), len(neg)
    pos_indices = np.random.choice(list(range(n)), min(n, sample_size), replace=False)
    neg_indices = np.random.choice(list(range(m)), min(m, sample_size), replace=False)
    return np.asarray(list(pos))[pos_indices, :], np.asarray(list(neg))[neg_indices, :]
