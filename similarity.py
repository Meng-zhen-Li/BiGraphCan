import numpy as np
import scipy.sparse as sp
import networkx as nx

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS
EMBEDDING_FILENAME = './embeddings.emb'

def similarity_matrix(adj, num_split):

    adj_ = adj
    adj_.data = np.ones(len(adj_.data))
    degrees = adj_.sum(axis=0)
    degrees[degrees == 1] = 0
    weights = sp.csr_matrix(1 / np.log10(degrees))
    A = adj_.multiply(weights) * adj_.T
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    adamic_adar = A

    common_neighbor = np.dot(adj, adj)

    alpha = 0.5
    S = sp.csr_matrix(np.sqrt(1 / np.sum(adj, axis=0))).multiply(adj)
    S = S.multiply(np.sqrt(1 / np.sum(adj, axis=1)))
    S.data[np.isnan(S.data)] = 0
    S.data[np.isinf(S.data)] = 0
    S2 = S
    A = alpha * S
    alpha2 = alpha
    for i in range(3):
        S2 = S2 * S
        alpha2 = alpha2 * alpha
        A = A + alpha2 * S2
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    von_neumann = A + A.transpose()

    alpha = 0.5
    D = sp.csr_matrix(np.sum(adj, axis=0))
    D[D == 0] = 1
    D.data = 1 / D.data
    S = D.multiply(adj)
    S.data[np.isnan(S.data)] = 0
    S.data[np.isinf(S.data)] = 0
    S2 = S
    A = alpha * S
    alpha2 = alpha
    for i in range(3):
        S2 = S2 * S
        alpha2 = alpha2 * alpha
        A = A + alpha2 * S2
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    rwr = A + A.transpose()

    similarities = [adamic_adar, rwr]
        
    similarities = [[s[:num_split, :num_split] for s in similarities], [s[num_split:, num_split:] for s in similarities]]
    similarities.append(adj + sp.eye(adj.shape[0]))

    return similarities