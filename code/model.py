# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
import pandas as pd
from utils_ import *
args = parameters_set()

# Attribute extractors
class MultiScaleCNN(nn.Module):
    def __init__(self, output_dim, vocab_size=5, embed_dim=64, num_filters=128, filter_sizes=(2, 3, 4), dropout=0.2):
        super(MultiScaleCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, kernel_size=fs) for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)

    def forward(self, seq_tensor):
        x = self.embedding(seq_tensor)  # (N, L, E)
        x = x.transpose(1, 2)  # (N, E, L)
        outs = []
        for conv in self.convs:
            c = F.relu(conv(x))  # (N, num_filters, L')
            p = F.max_pool1d(c, kernel_size=c.size(2)).squeeze(2)  # (N, num_filters)
            outs.append(p)
        out = torch.cat(outs, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class GATFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=4, dropout=0.2):
        super(GATFeatureExtractor, self).__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GATConv(input_dim, output_dim, heads=heads, concat=False, dropout=dropout))
        else:
            self.layers.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout))
            self.layers.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graphs):
        if isinstance(graphs, list):
            valid_graphs = [g for g in graphs if g is not None]
            if len(valid_graphs) == 0:
                # no valid graphs -> return empty tensor
                return torch.zeros((0, self.layers[-1].out_channels), device=torch.device('cpu'))
            graphs = Batch.from_data_list(valid_graphs)
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)
        pooled = global_mean_pool(x, batch) + global_max_pool(x, batch)
        return pooled


class FCNExtractor(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super(FCNExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, output_size)
        )

    def forward(self, x):
        return self.net(x)


# Graph-related classes
class MessageAggregator(nn.Module):
    def __init__(self, num_heads, hidden_size, attn_drop, alpha, name):
        super(MessageAggregator, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else nn.Identity()
        self.attn1 = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        self.attn2 = nn.Parameter(torch.empty(size=(1, self.num_heads, self.hidden_size)))
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        self.name = name

    def forward(self, nodes, metapath_instances, metapath_embedding, features):
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        num_nodes = len(nodes)
        device = metapath_embedding.device if isinstance(metapath_embedding, torch.Tensor) else next(self.parameters()).device
        h_out = torch.zeros(num_nodes, self.hidden_size * self.num_heads, device=device)

        indices_array = metapath_instances[self.name].values

        for node_idx, nid in enumerate(nodes):
            mask = (indices_array == nid)
            if not mask.any():
                continue
            # create boolean tensor on correct device to avoid device mismatch
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
            instance_indices = torch.where(mask_tensor)[0]  # on same device as metapath_embedding

            node_metapath_emb = metapath_embedding[instance_indices]  # (E, hidden_size * num_heads)

            if node_metapath_emb.dim() == 2:
                if node_metapath_emb.shape[1] == self.hidden_size * self.num_heads:
                    node_metapath_emb = node_metapath_emb.view(-1, self.num_heads, self.hidden_size)
                else:
                    node_metapath_emb = torch.cat([node_metapath_emb] * self.num_heads, dim=1)
                    node_metapath_emb = node_metapath_emb.view(-1, self.num_heads, self.hidden_size)

            node_feat = features[nid:nid + 1].to(device)  # ensure on same device
            node_feat_expanded = node_feat.expand(len(instance_indices), -1)

            a1 = self.attn1(node_feat_expanded)  # (E, num_heads)
            a2 = (node_metapath_emb * self.attn2).sum(dim=-1)  # (E, num_heads)
            a = self.leaky_relu(a1 + a2).unsqueeze(-1)  # (E, num_heads, 1)

            attention = F.softmax(a, dim=0)
            attention = self.attn_drop(attention)

            weighted = (attention * node_metapath_emb).sum(dim=0)  # (num_heads, hidden_size)

            h = F.elu(weighted).view(-1)
            h_out[node_idx] = h

        return h_out


class Subgraph_Fusion(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Subgraph_Fusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, z):
        w = self.project(z).mean(0)
        beta_ = torch.softmax(w, dim=0)
        beta = beta_.expand((z.shape[0],) + beta_.shape)
        return (beta * z).sum(1), beta_


class SemanticEncoder(nn.Module):
    def __init__(self, layer_num_heads, hidden_size, r_vec, etypes):
        super(SemanticEncoder, self).__init__()
        self.num_heads = layer_num_heads
        self.hidden_size = hidden_size
        self.r_vec = r_vec
        self.etypes = etypes

    def forward(self, edata):
        edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
        final_r_vec = torch.zeros([edata.shape[1], self.hidden_size // 2, 2], device=edata.device)
        r_vec = F.normalize(self.r_vec, p=2, dim=2)
        r_vec = torch.stack((r_vec, r_vec), dim=1)
        r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
        r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)
        final_r_vec[-1, :, 0] = 1
        for i in range(final_r_vec.shape[0] - 2, -1, -1):
            if self.etypes[i] is not None:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
            else:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
        for i in range(edata.shape[1] - 1):
            temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
            temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
            edata[:, i, :, 0] = temp1
            edata[:, i, :, 1] = temp2
        edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
        metapath_embedding = torch.mean(edata, dim=1)
        return metapath_embedding


class CMDKF_Layer(nn.Module):
    def __init__(self, meta_paths, test_data, hidden_size, r_vec, layer_num_heads, dropout, etypes, name):
        super(CMDKF_Layer, self).__init__()
        self.num_heads = layer_num_heads
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.r_vec = r_vec
        self.etypes = etypes
        self.message_aggregator_layer = nn.ModuleList()
        self.semantic_encoder_layer = nn.ModuleList()
        self.hidden_size = hidden_size
        self.test_data = test_data

        for i in range(len(meta_paths)):
            self.semantic_encoder_layer.append(
                SemanticEncoder(self.num_heads, self.hidden_size, self.r_vec, self.etypes[i]))

        for i in name:
            self.message_aggregator_layer.append(
                MessageAggregator(self.num_heads, self.hidden_size, attn_drop=dropout, alpha=0.01, name=i))

        self.subgraph_fusion = Subgraph_Fusion(in_size=self.hidden_size * self.num_heads)
        self.separate_metapath_subgraph = Separate_subgraph()
        self.exclude_test = Prevent_leakage(self.test_data)

    def stack_embedding(self, embeddings):
        subgraph_num_nodes = [embeddings[i].size()[0] for i in range(len(embeddings))]
        if subgraph_num_nodes.count(subgraph_num_nodes[0]) == len(subgraph_num_nodes):
            embeddings = torch.stack(embeddings, dim=1)
        else:
            for i in range(0, len(embeddings)):
                index = max(subgraph_num_nodes) - subgraph_num_nodes[i]
                if index != 0:
                    # create padding on same device as embeddings[i]
                    h_ = torch.zeros(index, self.hidden_size * self.num_heads, device=embeddings[i].device)
                    embeddings[i] = torch.cat((embeddings[i], h_), dim=0)
            embeddings = torch.stack(embeddings, dim=1)
        return embeddings

    def generate_metapath_instances(self, g, meta_path):
        edges = [g.edges(etype=f"{meta_path[j]}_{meta_path[j + 1]}") for j in range(len(meta_path) - 1)]
        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in range(len(edges))]
        df_0 = pd.DataFrame(edges[0], index=list(meta_path)[:2]).T
        df_1 = pd.DataFrame(edges[1], index=list(meta_path)[-2:]).T
        metapath_instances = pd.merge(df_0, df_1, how='inner')
        filt_metapath_instances = metapath_instances[['mi', 'dr', 'di']]
        filt_metapath_instances = self.exclude_test(filt_metapath_instances)
        metapath_instances = filt_metapath_instances[list(meta_path)]
        return metapath_instances

    def forward(self, g, h):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = self.separate_metapath_subgraph(g, meta_path)

        semantic_embeddings = {'mi': [], 'dr': [], 'di': []}
        nodes_embeddings = {}
        for i, meta_path in enumerate(self.meta_paths):
            edata_list = []
            new_g = self._cached_coalesced_graph[meta_path]
            metapath_instances = self.generate_metapath_instances(new_g, meta_path)
            for j in range(len(meta_path)):
                col_vals = metapath_instances.iloc[:, j].values
                idx = torch.tensor(col_vals, dtype=torch.long, device=h[list(meta_path)[j]].device)
                edata_list.append(F.embedding(idx, h[list(meta_path)[j]]).unsqueeze(1))
            edata = torch.hstack(edata_list)

            metapathembedding = self.semantic_encoder_layer[i](edata)

            nodes_mi = new_g.nodes('mi')
            nodes_dr = new_g.nodes('dr')
            nodes_di = new_g.nodes('di')
            if isinstance(nodes_mi, torch.Tensor):
                nodes_mi = nodes_mi.tolist()
            if isinstance(nodes_dr, torch.Tensor):
                nodes_dr = nodes_dr.tolist()
            if isinstance(nodes_di, torch.Tensor):
                nodes_di = nodes_di.tolist()

            semantic_embeddings['mi'].append(
                self.message_aggregator_layer[0](nodes_mi, metapath_instances, metapathembedding,
                                                 h['mi']))
            semantic_embeddings['dr'].append(
                self.message_aggregator_layer[1](nodes_dr, metapath_instances, metapathembedding,
                                                 h['dr']))
            semantic_embeddings['di'].append(
                self.message_aggregator_layer[2](nodes_di, metapath_instances, metapathembedding,
                                                 h['di']))

        for ntype in semantic_embeddings.keys():
            if ntype == 'mi':
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], mi_beta = self.subgraph_fusion(semantic_embeddings[ntype])
            elif ntype == 'dr' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], dr_beta = self.subgraph_fusion(semantic_embeddings[ntype])
            elif ntype == 'di' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], di_beta = self.subgraph_fusion(semantic_embeddings[ntype])

        return nodes_embeddings

class CMDKF(nn.Module):
    def __init__(self, meta_paths, test_data, in_size, hidden_size, num_heads, dropout, etypes):
        super(CMDKF, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.fc_mi = nn.Linear(in_size['mi'], hidden_size)
        self.fc_dr = nn.Linear(in_size['dr'], hidden_size)
        self.fc_di = nn.Linear(in_size['di'], hidden_size)

        self.mi_seq_extractor = MultiScaleCNN(output_dim=hidden_size)
        self.dr_graph_extractor = GATFeatureExtractor(input_dim=69, hidden_dim=64, output_dim=hidden_size)
        self.di_extractor = FCNExtractor(input_size=92, output_size=hidden_size)  # MDR 81, MDS 92

        self.predict = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_heads * 3, self.hidden_size * self.num_heads),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * self.num_heads, self.hidden_size * self.num_heads // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * self.num_heads // 4, self.hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1)
        )

        r_vec = nn.Parameter(torch.empty(size=(3, self.hidden_size // 2, 2)))

        self.layers1 = CMDKF_Layer(meta_paths, test_data, hidden_size, r_vec, num_heads, dropout, etypes,
                                    name=['mi', 'dr', 'di'])

        self.predict.apply(self.weights_init)
        nn.init.xavier_normal_(self.fc_mi.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_dr.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_di.weight, gain=1.414)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)

    def get_embed_map(self, attr_features, topo_features, data, fusion_weight):
        device = next(self.parameters()).device
        data_t = torch.as_tensor(data[:, :3], dtype=torch.long, device=device)

        out = []
        for col, key in enumerate(["mi", "dr", "di"]):
            ids = data_t[:, col]
            attr = attr_features[key][ids]
            if self.num_heads > 1:
                attr = attr.repeat(1, self.num_heads)

            topo_emb_all = topo_features[key]

            max_id = topo_emb_all.size(0)
            valid = (ids >= 0) & (ids < max_id)
            ids_safe = ids.clamp(0, max_id - 1)

            topo_emb = topo_emb_all[ids_safe]
            fused = fusion_weight * topo_emb + (1 - fusion_weight) * attr
            fused = torch.where(valid.unsqueeze(1), fused, attr)
            out.append(fused)

        return torch.cat(out, dim=1)


    def forward(self, g, inputs, att_input, data, fuse_weight):

        h_trans = {}
        h_trans['mi'] = F.elu(self.fc_mi(inputs['mi'])).view(-1, self.hidden_size)
        h_trans['dr'] = F.elu(self.fc_dr(inputs['dr'])).view(-1, self.hidden_size)
        h_trans['di'] = F.elu(self.fc_di(inputs['di'])).view(-1, self.hidden_size)

        h_trans = {k: v.to(next(self.parameters()).device) for k, v in h_trans.items()}

        h_trans_embed = self.layers1(g, h_trans)
        # print("h_trans_embed[key] shape: ", h_trans_embed['mi'].shape, h_trans_embed['dr'].shape,
        #       h_trans_embed['di'].shape)

        mi_seq = att_input['mi_seq']
        mi_attr_embed = self.mi_seq_extractor(mi_seq) if mi_seq is not None else torch.zeros((0, self.hidden_size), device=next(self.parameters()).device)

        dr_graphs = att_input['dr_graphs']
        dr_attr_embed = self.dr_graph_extractor(dr_graphs) if dr_graphs is not None else torch.zeros((0, self.hidden_size), device=next(self.parameters()).device)

        di_input_feat = att_input['di_attr']
        di_attr_embed = self.di_extractor(di_input_feat) if di_input_feat is not None else torch.zeros((0, self.hidden_size), device=next(self.parameters()).device)

        attr_embed = {'mi': mi_attr_embed, 'dr': dr_attr_embed, 'di': di_attr_embed}  # ,64
        # print("attr_embed[key] shape: ", attr_embed['mi'].shape, attr_embed['dr'].shape,
        #       attr_embed['di'].shape)
        h_concat = self.get_embed_map(attr_embed, h_trans_embed, data, fuse_weight)

        predict_score = torch.sigmoid(self.predict(h_concat))

        return predict_score