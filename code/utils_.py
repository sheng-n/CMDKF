"This code is based on HCMGNN"

import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import argparse


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.register_buffer('eps', torch.tensor(1e-5, dtype=torch.float))

    def forward(self, iput, target, gamma):

        batch_size = iput.size(0)
        loss_sum = torch.pow((iput - target), 2)
        result = (1.0 - gamma) * ((target * loss_sum).sum()) + gamma * (((1.0 - target) * loss_sum).sum())

        return result + self.eps

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)


class Matrix(nn.Module):

    def __init__(self):
        super(Matrix, self).__init__()

    def hits(self, pos_index, scores_index):

        if pos_index[0] in scores_index:
            Hits = 1
        else:
            Hits = 0
        return Hits

    def ndcg(self, pos_index, scores_index, n):

        dcg_sum = 0
        idcg_sum = 0
        for j in range(len(scores_index)):
            if scores_index[j] == pos_index[0]:
                dcg_sum += self.dcg(1, j + 1)
            else:
                dcg_sum += self.dcg(0, j + 1)

        for m in range(n):
            if m == 0:
                idcg_sum += self.dcg(1, m + 1)
            else:
                idcg_sum += self.dcg(0, m + 1)

        return dcg_sum, idcg_sum

    def dcg(self, rel, index):
        dcg = (2 ** rel - 1) / np.log2(index + 1)
        return dcg

    def forward(self, n, num, predict_val, num_pos, index):

        sample_hit, sample_ndcg = [], []
        Hits_sum = 0
        ndcg_sum = 0
        index_tuple = sorted(enumerate(index), reverse=False, key=lambda index: index[1])
        index_list = [index[0] for index in index_tuple]

        predict_val = predict_val[index_list]
        for i in range(num_pos):
            neg_scores = predict_val[num_pos + i * (num):num_pos + (i + 1) * (num)]
            scores = neg_scores.tolist() + [predict_val[i]]
            random_num = np.random.choice(range(len(scores)), len(scores), replace=False)
            pos_index = np.where(random_num == num)
            scores = np.array(scores)[random_num]
            scores_tuple = sorted(enumerate(scores), reverse=True, key=lambda scores: scores[1])
            scores_index = [scores[0] for scores in scores_tuple][:n]
            Hits = self.hits(pos_index, scores_index)
            dcg_sum, idcg_sum = self.ndcg(pos_index, scores_index, n)
            ndcg_sum += dcg_sum / idcg_sum
            Hits_sum += Hits
            sample_hit.append(Hits)
            sample_ndcg.append(dcg_sum / idcg_sum)
        Hits = Hits_sum / num_pos
        ndcg = ndcg_sum / num_pos
        return Hits, ndcg, sample_hit, sample_ndcg


class MRR(nn.Module):
    def __init__(self):
        super(MRR, self).__init__()

    def forward(self, num, predict_val, num_pos, index):
        sample_mrr = []
        rank_sum = 0
        index_tuple = sorted(enumerate(index), reverse=False, key=lambda index: index[1])
        index_list = [index[0] for index in index_tuple]
        predict_val = predict_val[index_list]
        for i in range(num_pos):
            neg_scores = predict_val[num_pos + i * (num):num_pos + (i + 1) * (num)]
            scores = neg_scores.tolist() + [predict_val[i]]
            random_num = np.random.choice(range(len(scores)), len(scores), replace=False)
            pos_index = np.where(random_num == num)
            scores = np.array(scores)[random_num]
            scores_tuple = sorted(enumerate(scores), reverse=True, key=lambda scores: scores[1])
            scores_index = [scores[0] for scores in scores_tuple]
            sample_mrr.append(1 / (scores_index.index(pos_index[0]) + 1))
            rank_sum += 1 / (scores_index.index(pos_index[0]) + 1)
        mrr = rank_sum / num_pos
        return mrr, sample_mrr


class HeteroGraph:
    def __init__(self, graph_dict):

        self._graph = {}
        self.canonical_etypes = []
        self._node_max = {}
        for key, (srcs, dsts) in graph_dict.items():
            self.canonical_etypes.append(key)
            src_tensor = torch.LongTensor(srcs) if not isinstance(srcs, torch.Tensor) else srcs.clone().long()
            dst_tensor = torch.LongTensor(dsts) if not isinstance(dsts, torch.Tensor) else dsts.clone().long()
            self._graph[key] = (src_tensor, dst_tensor)

            if src_tensor.numel() > 0:
                self._node_max.setdefault(key[0], -1)
                self._node_max[key[0]] = max(self._node_max[key[0]], int(src_tensor.max().item()))
            if dst_tensor.numel() > 0:
                self._node_max.setdefault(key[2], -1)
                self._node_max[key[2]] = max(self._node_max[key[2]], int(dst_tensor.max().item()))

        for ntype, mx in list(self._node_max.items()):
            self._node_max[ntype] = mx + 1 if mx >= 0 else 0

    def edges(self, etype):

        for c in self.canonical_etypes:
            if c[1] == etype:
                return self._graph[c]
        return torch.LongTensor([]), torch.LongTensor([])

    def nodes(self, ntype):

        cnt = self._node_max.get(ntype, 0)
        return torch.arange(cnt, dtype=torch.long) if cnt > 0 else torch.LongTensor([])

    def to(self, device):

        for k, (src, dst) in list(self._graph.items()):
            if isinstance(src, torch.Tensor):
                src = src.to(device)
            if isinstance(dst, torch.Tensor):
                dst = dst.to(device)
            self._graph[k] = (src, dst)
        return self

    def __repr__(self):
        return f"HeteroGraph(canonical_etypes={self.canonical_etypes})"


def construct_hg(pos_data):

    mi_dr_edges = pos_data[:, [0, 1]]
    dr_di_edges = pos_data[:, [1, 2]]
    mi_di_edges = pos_data[:, [0, 2]]

    mi_dr_edges = np.unique(mi_dr_edges, axis=0)
    dr_di_edges = np.unique(dr_di_edges, axis=0)
    mi_di_edges = np.unique(mi_di_edges, axis=0)

    mi_dr_edges = mi_dr_edges[mi_dr_edges[:, 0].argsort()]
    dr_di_edges = dr_di_edges[dr_di_edges[:, 0].argsort()]
    mi_di_edges = mi_di_edges[mi_di_edges[:, 0].argsort()]

    graph_data = {

        ('mi', 'mi_dr', 'dr'): (mi_dr_edges[:, 0].tolist(), mi_dr_edges[:, 1].tolist()),
        ('dr', 'dr_di', 'di'): (dr_di_edges[:, 0].tolist(), dr_di_edges[:, 1].tolist()),
        ('mi', 'mi_di', 'di'): (mi_di_edges[:, 0].tolist(), mi_di_edges[:, 1].tolist()),

        ('dr', 'dr_mi', 'mi'): (mi_dr_edges[:, 1].tolist(), mi_dr_edges[:, 0].tolist()),
        ('di', 'di_dr', 'dr'): (dr_di_edges[:, 1].tolist(), dr_di_edges[:, 0].tolist()),
        ('di', 'di_mi', 'mi'): (mi_di_edges[:, 1].tolist(), mi_di_edges[:, 0].tolist())
    }
    return HeteroGraph(graph_data)

class Prevent_leakage(nn.Module):
    def __init__(self, test_data):
        super(Prevent_leakage, self).__init__()
        self.test_data = test_data

    def forward(self, metapath_instances):
        test_pos_data = pd.DataFrame(self.test_data[:, :3], columns=['mi', 'dr', 'di'])
        metapath_instances_all = pd.concat([metapath_instances, test_pos_data, test_pos_data], ignore_index=True)
        exclude_metapath_instances = metapath_instances_all.drop_duplicates(subset=['mi', 'dr', 'di'], keep=False)
        exclude_metapath_instances = exclude_metapath_instances.reset_index(drop=True)
        return exclude_metapath_instances


class Separate_subgraph(nn.Module):
    def __init__(self):
        super(Separate_subgraph, self).__init__()

    def get_edges(self, edges1, edges2):
        new_edges = [[list() for j in range(2)] for i in range(2)]
        for i in range(len(edges1[0])):
            if edges1[1][i] in edges2[0]:
                new_edges[0][0].append(edges1[0][i])
                new_edges[0][1].append(edges1[1][i])
                index = [m for m, x in enumerate(edges2[0]) if x == edges1[1][i]]
                if edges1[1][i] not in new_edges[1][0]:
                    for j in range(len(index)):
                        new_edges[1][0].append(edges1[1][i])
                        new_edges[1][1].append(edges2[1][index[j]])
        return new_edges

    def forward(self, hg, metapath):

        new_triplets_edge = []
        metapath_list = [f"{metapath[i]}_{metapath[i + 1]}" for i in range(len(metapath) - 1)]
        edges = [hg.edges(etype=metapath_list[i]) for i in range(len(metapath_list))]

        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in range(len(edges))]
        if len(metapath_list) == 2:
            new_edges = self.get_edges(edges[0], edges[1])
        elif len(metapath_list) == 3:
            new_edges = self.get_edges(edges[0], edges[1])
            new_edges1 = self.get_edges(new_edges[1], edges[2])
            new_edges.append(new_edges1[1])
        for path in metapath_list:
            for i in range(len(hg.canonical_etypes)):
                if path in hg.canonical_etypes[i]:
                    new_triplets_edge.append(hg.canonical_etypes[i])
        graph_data = {}
        for i in range(len(metapath_list)):

            graph_data[new_triplets_edge[i]] = (new_edges[i][0], new_edges[i][1])

        subgraph = HeteroGraph(graph_data)
        return subgraph


def ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix, epoch_max_matrix, e, hits_1,
              hits_3, hits_5, ndcg1, ndcg3, ndcg5, MRR):
    if hits_1 >= hits_max_matrix[0][0]:
        hits_max_matrix[0][0] = hits_1
        hits_max_matrix[0][1] = hits_3
        hits_max_matrix[0][2] = hits_5
        NDCG_max_matrix[0][0] = ndcg1
        NDCG_max_matrix[0][1] = ndcg3
        NDCG_max_matrix[0][2] = ndcg5
        MRR_max_matrix[0][0] = MRR
        epoch_max_matrix[0][0] = e
        patience_num_matrix[0][0] = 0
    else:
        patience_num_matrix[0][0] += 1
    return patience_num_matrix


def parameters_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=111, help='the random seeds')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--loss_gamma', type=float, default=0.4, help="balance coefficient")
    parser.add_argument('--hidden_size', type=int, default=128, help='model hidden dim')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads of GAT')
    parser.add_argument('--fuse_weight', type=float, default=0.4, help='View fusion weight')
    parser.add_argument('--patience', type=int, default=50, help='number of patience of early stopping mechanism')
    parser.add_argument('--etypes', nargs='+', default=[[0, 1], [2, 3], [4, 0], [5, 2], [3, 5], [1, 4]],
                        help='the types of edges contained in the six metapaths')

    parser.add_argument('--metapaths', nargs='+',
                        default=[['mi', 'dr', 'di'], ['dr', 'mi', 'di'], ['di', 'mi', 'dr'], ['di', 'dr', 'mi'], ['mi', 'di', 'dr'],
                                 ['dr', 'di', 'mi']],
                        help='the types of metapaths')

    # domain knowledge
    parser.add_argument('--miRNA_feature', default="MDS_data/mi_kmer.txt",
                        help='miRNA feature')

    parser.add_argument('--drug_feature', default="MDS_data/drug_maccs.txt",
                        help='drug feature')

    parser.add_argument('--disease_feature', default="MDS_data/dis_sim.txt",
                        help='disease feature')

    # feature
    parser.add_argument('--miRNA_sequence', default="MDS_data/miRNA_sequence.xlsx",
                        help='miRNA sequence')

    parser.add_argument('--drug_smiles', default="MDS_data/drug_smiles.xlsx",
                        help='drug smiles')

    parser.add_argument('--disease_similarity', default="MDS_data/dis_sim.txt",
                        help='disease similarity')

    # interactions
    parser.add_argument('--cv_data', default="MDS_data/CV_data/CV_",
                        help='Path to cv data.')

    parser.add_argument('--indepent_data', default="MDS_data/indepent_data",
                        help='Path to indepent data. ')

    args = parser.parse_args()

    return args