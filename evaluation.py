import numpy as np
import networkx as nx
import random
import scipy.sparse as sp
import copy
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

def precision_at_k_score(y_true, y_pred_proba, k=1000, pos_label=1):
    topk = [y_true_ == pos_label for y_true_, y_pred_proba_ in sorted(
        zip(y_true, y_pred_proba), key=lambda y: y[1], reverse=True)[:k]]
    return sum(topk) / len(topk)


def LinkPrediction(embedding_look_up, edgelist_bipartite, testing_nodes):
    num_u = len(np.unique(edgelist_bipartite[:, 0]))
    num_v = len(np.unique(edgelist_bipartite[:, 1]))

    """ indices = []
    indices.extend(range(50))
    indices.extend(range(num_u, num_u + 50))
    for i in indices: """
    for i in range(len(testing_nodes)):
        node = testing_nodes[i]
        idx = np.where(np.any(edgelist_bipartite != node,axis = 1))
        train_pos_edges = edgelist_bipartite[idx[0]]
        idx = np.where(np.any(edgelist_bipartite == node,axis = 1))
        test_pos_edges = edgelist_bipartite[idx[0]]
        if i < num_u:
            test_neg_edges = np.append(np.ones((num_v - len(test_pos_edges), 1)) * node, np.reshape(np.setdiff1d(np.unique(edgelist_bipartite[:, 1]), test_pos_edges[:, 1]), (-1, 1)), axis=1)
        else:
            test_neg_edges = np.append(np.reshape(np.setdiff1d(np.unique(edgelist_bipartite[:, 0]), test_pos_edges[:, 0]), (-1, 1)), np.ones((num_u - len(test_pos_edges), 1)) * node, axis=1)
        test_neg_edges = test_neg_edges.astype(int)
        test_neg_edges = test_neg_edges[random.sample(range(len(test_neg_edges)), len(test_pos_edges))]

        # construct X_train, y_train, X_test, y_test
        X_train = []
        y_train = []
        for edge in train_pos_edges:
            node_u_emb = embedding_look_up[edge[0]]
            node_v_emb = embedding_look_up[edge[1]]
            feature_vector = np.append(node_u_emb, node_v_emb)
            X_train.append(feature_vector)
            y_train.append(1)
        
        train_neg_edges = np.append(random.choices([[x] for x in np.unique(train_pos_edges[:, 0])], k=len(train_pos_edges)), random.choices([[x] for x in np.unique(train_pos_edges[:, 1])], k=len(train_pos_edges)), 1)
        for edge in train_neg_edges:
            node_u_emb = embedding_look_up[edge[0]]
            node_v_emb = embedding_look_up[edge[1]]
            feature_vector = np.append(node_u_emb, node_v_emb)
            X_train.append(feature_vector)
            y_train.append(0)

        X_test = []
        y_test = []
        for edge in test_pos_edges:
            node_u_emb = embedding_look_up[edge[0]]
            node_v_emb = embedding_look_up[edge[1]]
            feature_vector = np.append(node_u_emb, node_v_emb)
            X_test.append(feature_vector)
            y_test.append(1)

        for edge in test_neg_edges:
            node_u_emb = embedding_look_up[edge[0]]
            node_v_emb = embedding_look_up[edge[1]]
            feature_vector = np.append(node_u_emb, node_v_emb)
            X_test.append(feature_vector)
            y_test.append(0)

        # shuffle for training and testing
        c = list(zip(X_train, y_train))
        random.shuffle(c)
        X_train, y_train = zip(*c)

        c = list(zip(X_test, y_test))
        random.shuffle(c)
        X_test, y_test = zip(*c)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        clf1 = LogisticRegression(random_state=None, solver='lbfgs')
        clf1.fit(X_train, y_train)
        y_pred_proba = clf1.predict_proba(X_test)[:, 1]
        y_pred = clf1.predict(X_test)
        f = open('results.txt', 'a')

        """ precision_at_k = precision_at_k_score(y_test, y_pred_proba, len(test_pos_edges))
        f.write(f'{precision_at_k:.3f}')
        """
        auc = roc_auc_score(y_test, y_pred_proba)
        aupr = average_precision_score(y_test, y_pred_proba)
        f.write(f'({auc:.3f}, {aupr:.3f})')
        """
        ranking = np.argsort(np.argsort(-y_pred_proba))
        idx = (y_test==1)
        ranking = ranking[idx]
        f.write(f'({np.min(ranking) / len(y_pred_proba):.3f}, {np.median(ranking) / len(y_pred_proba):.3f}, {np.max(ranking) / len(y_pred_proba):.3f})') """
        if i == num_u - 1 or i  == len(testing_nodes) - 1:
        # if i == 49 or i  == num_u + 49:
            f.write('\n')
        else:
            f.write(';')
    
    return
