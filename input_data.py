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
    testing_nodes = data['testing_nodes'].flatten()
    adj = sp.csr_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(
        np.max(edgelist) + 1, np.max(edgelist) + 1))
    adj = adj + adj.transpose()
    adj.data = np.ones(len(adj.data))
    idx = (edgelist[:, 0] < num_u) & (edgelist[:, 1] >= num_u)
    edgelist_bipartite = edgelist[idx[0]]
    idx = ((edgelist[:, 0] < num_u) & (edgelist[:, 1] < num_u)) | (
        (edgelist[:, 0] >= num_u) & (edgelist[:, 1] >= num_u))
    edgelist_no_bipartite = edgelist[idx[0]]

    adj_no_bipartite = sp.csr_matrix((np.ones(len(edgelist_no_bipartite)), (edgelist_no_bipartite[:, 0], edgelist_no_bipartite[:, 1])), shape=(np.max(edgelist) + 1, np.max(edgelist) + 1))
    adj_no_bipartite = adj_no_bipartite + adj_no_bipartite.transpose()
    adj_no_bipartite.data = np.ones(len(adj_no_bipartite.data))

    if 'features' in data.keys():
        features = data['features']
    else:
        features = sp.identity(adj.shape[0])

    return adj, int(num_u), features, adj_no_bipartite, edgelist_bipartite, testing_nodes


def perturb(edgelist, perturb_ratio):
    idx = (np.random.rand(len(edgelist)) > perturb_ratio)
    new_edgelist = edgelist[idx]
    return new_edgelist
