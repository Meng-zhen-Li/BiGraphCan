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
    num_u = data['num_u'][0][0]
    adj = sp.csr_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])), shape=(
        np.max(edgelist) + 1, np.max(edgelist) + 1))
    adj = adj + adj.transpose()
    adj.data = np.ones(len(adj.data))
    idx = (edgelist[:, 0] < num_u) & (edgelist[:, 1] >= num_u)
    edgelist_bipartite = edgelist[idx]
    testing_nodes = [int(x) for x in FLAGS.testing_nodes.split(',')]
    edgelist_train = edgelist
    if FLAGS.task == 'cold_start':
        if testing_nodes[0] <= np.max(edgelist_bipartite[:, 0]):
            mask = np.isin(edgelist_train[:, 0], testing_nodes)
            mask_constraint = edgelist_train[:, 1] >= num_u
        else:
            mask = np.isin(edgelist_train[:, 1], testing_nodes)
            mask_constraint = edgelist_train[:, 0] < num_u
        combined_mask = mask & mask_constraint
        edgelist_train = edgelist_train[~combined_mask]
    else:
        for testing_node in testing_nodes:
            if testing_node <= np.max(edgelist_bipartite[:, 0]):
                mask = (edgelist_train[:, 0] == testing_node) & (edgelist_train[:, 1] >= num_u)
            else:
                mask = (edgelist_train[:, 0] < num_u) & (edgelist_train[:, 1] == testing_node)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                nodes = np.unique(edgelist_train[idx, 1 if testing_node <= np.max(edgelist_bipartite[:, 0]) else 0])
                mask = np.isin(edgelist_train[:, 1 if testing_node <= np.max(edgelist_bipartite[:, 0]) else 0], nodes)
                mask = mask & (edgelist_train[:, 0] < num_u if testing_node <= np.max(edgelist_bipartite[:, 0]) else edgelist_train[:, 1] >= num_u)
                edgelist_train = edgelist_train[~mask]
        assert len(np.unique(edgelist_train)) == len(np.unique(edgelist))
    if FLAGS.noise > 0:
        f = open('noise.txt', 'a')
        idx = ((edgelist_train[:, 0] < num_u) & (edgelist_train[:, 1] < num_u)) | ((edgelist_train[:, 0] >= num_u) & (edgelist_train[:, 1] >= num_u))
        unique, counts = np.unique(edgelist_train[idx, :], return_counts=True)
        counts = dict(zip(unique, counts))
        idx = np.where(idx)[0]
        tried = 0.0
        successed = 0.0
        noise = []
        while successed < int(len(idx) * FLAGS.noise):
            i = idx[np.random.randint(len(idx))]
            if i not in noise and edgelist_train[i][0] != edgelist_train[i][1] and counts[edgelist_train[i][0]] > 1 and counts[edgelist_train[i][1]] > 1:
                noise.append(i)
                counts[edgelist_train[i][0]] = counts[edgelist_train[i][0]] - 1
                counts[edgelist_train[i][1]] = counts[edgelist_train[i][1]] - 1
                successed = successed + 1
            tried = tried + 1
            if tried > 100 and successed / tried < 0.5:
                break
        edgelist_train = np.delete(edgelist_train, noise, axis=0)
        f.write(f'{successed / len(idx):.3f}\n')
        assert len(np.unique(edgelist_train)) == len(np.unique(edgelist))
        f.close()
            

    adj_train = sp.csr_matrix((np.ones(len(edgelist_train)), (edgelist_train[:, 0], edgelist_train[:, 1])), shape=(np.max(edgelist) + 1, np.max(edgelist) + 1))
    adj_train = adj_train + adj_train.transpose()
    adj_train.data = np.ones(len(adj_train.data))

    if 'features' in data.keys():
        features = data['features']
    else:
        features = sp.identity(adj.shape[0])

    return adj, int(num_u), features, adj_train, edgelist_bipartite
