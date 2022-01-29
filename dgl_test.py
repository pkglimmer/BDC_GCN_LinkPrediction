import pandas as pd
import numpy as np
import dgl
from dgl.data import DGLDataset
import torch
import os
from torch_geometric.nn import GCNConv, GraphConv, GATConv, GatedGraphConv
import dgl.data
import dgl.function as fn
from dgl.nn.pytorch.conv.sageconv import SAGEConv
# from loss import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

data_dir  = './bill_challenge_datasets'
train_graph = pd.read_csv(f'{data_dir}/Training/training_graph.csv')
page_label = pd.read_csv(f'{data_dir}/Training/node_classification.csv')
iso_nodes = pd.read_csv(f'{data_dir}/Training/isolated_nodes.csv')
iso_nodes = list(iso_nodes.nodes)

class BDCDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='karate_club')

    def process(self):
        node_set = {1}
        for index, row in train_graph.iterrows():
            if row.node1 != row.node2:
                node_set.update([row.node1, row.node2])
        node_list = list(node_set)
        node_to_index  = {node: index for index, node in enumerate(node_list)}

        # construct edges (consecutive integers starting from 0)
        edges_src, edges_dst = [], []
        for index, row in train_graph.iterrows():
            if row.node1 != row.node2:
                edges_src.append(node_to_index[row.node1])
                edges_dst.append(node_to_index[row.node2])
        edges_src = torch.from_numpy(np.array(edges_src))
        edges_dst = torch.from_numpy(np.array(edges_dst))
        # construct labels and features
        node_feature = pd.read_csv('node_feature.csv')
        node_feature['d2v'] = node_feature['d2v'].apply(lambda x: np.fromstring(x.replace('\n','').replace('[','').replace(']','').replace('  ',' '), sep=' '))
        for index, row in node_feature.iterrows():
            try:
                node_feature.loc[index, 'id'] = node_to_index[node_feature.loc[index, 'id']]
            except:
                node_feature = node_feature.drop(index)
                
        node_labels = torch.from_numpy(node_feature['label'].to_numpy())
        node_d2v = torch.from_numpy(np.stack(list(node_feature['d2v']), axis=0))
                
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=len(node_list))
        self.graph.ndata['label'] = node_labels
        self.graph.ndata['feat'] = node_d2v
        # self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = len(node_list)
        n_train = int(n_nodes * 0.8)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

dataset = BDCDataset()
g = dataset[0]
g = dgl.to_bidirected(g, copy_ndata=True)
u, v = g.edges()

eids = np.arange(g.number_of_edges())  # [0 to num-1]
eids = np.random.permutation(eids)  # permutation
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]  # test set of pos edge
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]  # train set of pos edge

adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
train_g = dgl.remove_edges(g, eids[:test_size])
print('f=========================')

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

#
class DGLGCN(nn.Module):
    def __init__(self, reid_feature_dim, st_feature_dim, reid_nhid, st_nhid, nclass,
                 dropout=0., bn=False, gn=False, residual=False):
        super(DGLGCN, self).__init__()

        # BUILD CONV MODULE
        # in_feats, out_feats, aggregator_type,
        # feat_drop=0., bias=True, norm=None, activation=None
        self.reid_conv1 = GraphConv(reid_feature_dim, reid_nhid, 'mean')
        self.reid_conv2 = GraphConv(reid_nhid, reid_nhid, 'mean')

        self.st_conv1 = GraphConv(st_feature_dim, st_nhid, 'mean')
        self.st_conv2 = GraphConv(st_nhid, st_nhid, 'mean')

        out_size = (reid_nhid + st_nhid) / 2
        self.cat_conv1 = GraphConv(int(reid_nhid + st_nhid), int(out_size), 'mean')
        self.cat_conv2 = GraphConv(int(out_size), int(out_size), 'mean')

    # FORWARD
    # def forward(self, g, in_feat):
    def forward(self, g, data):

        reid_x, st_x, adj, idx = data[0], data[1], data[2], data[3]

        reid_x = self.conv1(g, reid_x)
        reid_x = F.relu(reid_x)
        reid_x = self.conv1(g, reid_x)

        st_x = self.conv1(g, st_x)
        st_x = F.relu(st_x)
        st_x = self.conv1(g, st_x)

        cat_feature = torch.cat([reid_x, st_x], dim=-1)  # cat app and　st feature
        cat_feature = self.cat_conv1(g, cat_feature)
        cat_feature = F.relu(cat_feature)
        cat_feature = self.cat_conv2(cat_feature)
        
        return cat_feature


# The model then predicts the probability of existence of an edge
# by computing a score between the representations of both incident
# nodes with a function (e.g. an MLP or a dot product),
# which you will see in the next section.

train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

import dgl.function as fn

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        
print('================Building model=================')
model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
# You can replace DotPredictor with MLPPredictor.
#pred = MLPPredictor(16)
pred = DotPredictor()

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
model = model.double()
for e in range(100):
    # forward
    h = model(train_g, train_g.ndata['feat'])
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))