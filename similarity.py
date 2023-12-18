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

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    DA = d_mat_inv_sqrt.dot(adj)
    resource_allocate = adj.dot(DA).tocoo()

    rowsum = np.array(adj.sum(1))
    deg_row = np.tile(rowsum, (1,adj.shape[0]))
    deg_row = sp.coo_matrix(deg_row)
    hdi = adj.dot(adj)
    X = hdi.astype(bool).astype(int)
    deg_row = deg_row.multiply(X)
    deg_row = mymaximum(deg_row, deg_row.T)
    hdi = hdi/deg_row
    whereAreNan = np.isnan(hdi)
    whereAreInf = np.isinf(hdi)
    hdi[whereAreNan] = 0
    hdi[whereAreInf] = 0
    hdi = sp.coo_matrix(hdi)

    hpi = adj.dot(adj)
    X = hpi.astype(bool).astype(int)
    deg_row = deg_row.multiply(X)
    deg_row = myminimum(deg_row, deg_row.T)
    hpi = hpi/deg_row
    whereAreNan = np.isnan(hpi)
    whereAreInf = np.isinf(hpi)
    hpi[whereAreNan] = 0
    hpi[whereAreInf] = 0
    hpi = sp.coo_matrix(hpi)

    similarities = [adamic_adar, common_neighbor, von_neumann, rwr, adj + sp.eye(adj.shape[0]), resource_allocate, hdi, hpi]

    similarities = [[s[:num_split, :num_split] for s in similarities], [s[num_split:, num_split:] for s in similarities]]

    return similarities

def mymaximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data <= 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def myminimum(A,B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data >= 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)