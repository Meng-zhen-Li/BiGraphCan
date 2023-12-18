import numpy as np
import random
import scipy.sparse as sp
from scipy.io import loadmat
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def load_data(datastr):
    data = loadmat('data/' + datastr + '.mat')
    edgelist = data['edgelist']
    num_u = data['num_u']
    adj = sp.csr_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(np.max(edgelist) + 1, np.max(edgelist) + 1))
    adj = adj + adj.transpose()
    adj.data = np.ones(len(adj.data))
    idx = (edgelist[:, 0] < num_u) & (edgelist[:, 1] >= num_u)
    edgelist_bipartite = edgelist[idx[0]]
    adj_train, training_neg_edges, testing_pos_edges, testing_neg_edges = split_train_test_graph(edgelist, edgelist_bipartite)
    
    if 'features' in data.keys():
        features = data['features']
    else:
        features = sp.identity(adj.shape[0])

    return adj, int(num_u), features, adj_train, training_neg_edges, testing_pos_edges, testing_neg_edges

def split_train_test_graph(edgelist, edgelist_bipartite, testing_ratio=0.2, seed=None):
    testing_edges_num = int(len(edgelist_bipartite) * testing_ratio)
    random.seed(seed)

    # Positive Edges
    testing_pos_edges = []
    initial_edgelist = edgelist
    while len(testing_pos_edges) < testing_edges_num:
        testing_idx = random.choice(range(len(edgelist_bipartite)))
        if np.count_nonzero(edgelist[:, 0] == edgelist_bipartite[testing_idx, 0]) > 1 and np.count_nonzero(edgelist[:, 1] == edgelist_bipartite[testing_idx, 1]) > 1:
            testing_pos_edges.append(edgelist_bipartite[testing_idx])
            idx = np.argwhere((edgelist[:, 0] == edgelist_bipartite[testing_idx][0]) & (edgelist[:, 1] == edgelist_bipartite[testing_idx][1]))
            edgelist_bipartite = np.delete(edgelist_bipartite, testing_idx, 0)
            edgelist = np.delete(edgelist, idx, 0)

    assert len(np.unique(edgelist[:, 0])) == len(np.unique(initial_edgelist[:, 0]))
    assert len(np.unique(edgelist[:, 1])) == len(np.unique(initial_edgelist[:, 1]))

    adj_train = sp.csr_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(np.max(edgelist) + 1, np.max(edgelist) + 1))
    adj_train = adj_train + adj_train.transpose()
    adj_train.data = np.ones(len(adj_train.data))

    # Negative Edges
    testing_neg_edges = np.append(random.choices([[x] for x in np.unique(edgelist_bipartite[:, 0])], k=testing_edges_num), random.choices([[x] for x in np.unique(edgelist_bipartite[:, 1])], k=testing_edges_num), 1)
    training_neg_edges = np.append(random.choices([[x] for x in np.unique(edgelist_bipartite[:, 0])], k=len(edgelist_bipartite)), random.choices([[x] for x in np.unique(edgelist_bipartite[:, 1])], k=len(edgelist_bipartite)), 1)

    return adj_train, training_neg_edges, testing_pos_edges, testing_neg_edges