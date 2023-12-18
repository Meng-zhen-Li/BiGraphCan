import numpy as np
import networkx as nx
import random
import copy
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def precision_at_k_score(y_true, y_pred_proba, k=1000, pos_label=1):
    topk = [y_true_ == pos_label for y_true_, y_pred_proba_ in sorted(zip(y_true, y_pred_proba), key=lambda y: y[1], reverse=True)[:k]]
    return sum(topk) / len(topk)

def LinkPrediction(embedding_look_up, train_graph, train_neg_edges, test_pos_edges, test_neg_edges, seed=None):
    random.seed(seed)

    # construct X_train, y_train, X_test, y_test
    X_train = []
    y_train = []
    for edge in train_graph.edges():
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(1)
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

    clf1 = LogisticRegression(random_state=seed, solver='lbfgs', max_iter=1000)
    clf1.fit(X_train, y_train)
    y_pred_proba = clf1.predict_proba(X_test)[:, 1]
    y_pred = clf1.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f = open('results.txt', 'a')

    """ precision_at_k = [precision_at_k_score(y_test, y_pred_proba, x) for x in [1, 10, 100, 1000, 10000]]
    for score in precision_at_k:
        f.write(f'{score:.3f},')
    f.write('\n') """

    f.write('#' * 9 + ' Link Prediction Performance ' + '#' * 9 + '\n')
    f.write(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}' + '\n')
    f.write('#' * 50 + '\n' + '\n')
    f.close()
    return auc_roc, auc_pr