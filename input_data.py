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
    adj = sp.csr_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(
        np.max(edgelist) + 1, np.max(edgelist) + 1))
    adj = adj + adj.transpose()
    adj.data = np.ones(len(adj.data))
    idx = (edgelist[:, 0] < num_u) & (edgelist[:, 1] >= num_u)
    edgelist_bipartite = edgelist[idx[0]]
    testing_node = int(FLAGS.testing_node)
    if testing_node <= np.max(edgelist_bipartite[:, 0]):
        idx = ~((edgelist[:, 0] == testing_node) & (edgelist[:, 1] >= num_u))
    else:
        idx = ~((edgelist[:, 0] < num_u) & (edgelist[:, 1] == testing_node))
    edgelist_train = edgelist[idx[0]]

    adj_train = sp.csr_matrix((np.ones(len(edgelist_train)), (edgelist_train[:, 0], edgelist_train[:, 1])), shape=(np.max(edgelist) + 1, np.max(edgelist) + 1))
    adj_train = adj_train + adj_train.transpose()
    adj_train.data = np.ones(len(adj_train.data))

    if 'features' in data.keys():
        features = data['features']
    else:
        features = sp.identity(adj.shape[0])

    return adj, int(num_u), features, adj_train, edgelist_bipartite
