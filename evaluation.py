import numpy as np
import networkx as nx
import random
import scipy.sparse as sp
import copy
import itertools

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

def precision_at_k_score(y_true, y_pred_proba, k=1000, pos_label=1):
    topk = [y_true_ == pos_label for y_true_, y_pred_proba_ in sorted(
        zip(y_true, y_pred_proba), key=lambda y: y[1], reverse=True)[:k]]
    return sum(topk) / len(topk)


def generate_neg_edges(original_graph, testing_edges_num, num_split, seed):
    L = list(original_graph.nodes())

    # create a complete graph
    G = nx.Graph()
    G.add_nodes_from(L)
    G.add_edges_from(itertools.combinations(L, 2))
    # remove original edges
    G.remove_edges_from(original_graph.edges())
    random.seed(seed)
    neg_edges = random.sample([list(x) for x in list(G.edges()) if x[0] <= num_split and x[1] > num_split or x[0] > num_split and x[1] <= num_split], testing_edges_num)
    neg_edges = [sorted(x) for x in neg_edges]
    return neg_edges


def LinkPrediction(embedding_look_up, edgelist_bipartite, testing_nodes):
    num_u = len(np.unique(edgelist_bipartite[:, 0]))
    num_v = len(np.unique(edgelist_bipartite[:, 1]))

    test_pos_edges = []
    test_neg_edges = []
    for testing_node in testing_nodes:
        idx = np.where(np.any(edgelist_bipartite != testing_node,axis = 1))
        train_pos_edges = edgelist_bipartite[idx[0]]
        idx = np.where(np.any(edgelist_bipartite == testing_node,axis = 1))
        pos_edges = edgelist_bipartite[idx[0]]
        if testing_node <= np.max(edgelist_bipartite[:, 0]):
            neg_edges = np.append(np.ones((num_v - len(pos_edges), 1)) * testing_node, np.reshape(np.setdiff1d(np.unique(edgelist_bipartite[:, 1]), pos_edges[:, 1]), (-1, 1)), axis=1)
        else:
            neg_edges = np.append(np.reshape(np.setdiff1d(np.unique(edgelist_bipartite[:, 0]), pos_edges[:, 0]), (-1, 1)), np.ones((num_u - len(pos_edges), 1)) * testing_node, axis=1)
        neg_edges = neg_edges.astype(int)
        idx = random.sample(list(range(len(neg_edges))), len(pos_edges))
        if test_pos_edges == []:
            test_pos_edges = pos_edges
        else:
            test_pos_edges = np.append(test_pos_edges, pos_edges, axis=0)
        if test_neg_edges == []:
            test_neg_edges = neg_edges[idx, :]
        else:
            test_neg_edges = np.append(test_neg_edges, neg_edges[idx, :], axis=0)

    # construct X_train, y_train, X_test, y_test
    X_train = []
    y_train = []
    for edge in train_pos_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(1)
    
    train_neg_edges = generate_neg_edges(nx.from_edgelist(edgelist_bipartite), len(train_pos_edges), np.max(edgelist_bipartite[:, 0]), None)
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

    # clf1 = LogisticRegression(random_state=None, solver='lbfgs', max_iter=1000)
    clf1 = XGBClassifier(learning_rate= 0.4, max_depth= 4, min_child_weight=2)
    clf1.fit(X_train, y_train)
    y_pred_proba = clf1.predict_proba(X_test)[:, 1]
    f = open('results.txt', 'a')

    """ precision_at_k = []
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        precision_at_k.append(str(precision_at_k_score(y_test, y_pred_proba, i)))
    precision_at_k = ','.join(precision_at_k)
    f.write(f'{precision_at_k};') """

    auc = roc_auc_score(y_test, y_pred_proba)
    aupr = average_precision_score(y_test, y_pred_proba)
    precision_at_k = precision_at_k_score(y_test, y_pred_proba, 100)
    f.write(f'({precision_at_k:.3f},{auc:.3f}, {aupr:.3f})')
    f.write(';')
    
    return

def ZeroShot(embedding_look_up, edgelist_bipartite, testing_nodes):
    num_u = len(np.unique(edgelist_bipartite[:, 0]))
    num_v = len(np.unique(edgelist_bipartite[:, 1]))

    train_pos_edges = edgelist_bipartite
    test_pos_edges = np.empty((0, 2))
    for node in testing_nodes:
        idx = np.where(train_pos_edges==node)
        test_pos_edges = np.append(test_pos_edges, train_pos_edges[idx[0]], axis=0)
        train_pos_edges = np.delete(train_pos_edges, idx[0], axis=0)

    # construct X_train, y_train
    X_train = []
    y_train = []
    for edge in train_pos_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(1)
    
    train_neg_edges = generate_neg_edges(nx.from_edgelist(edgelist_bipartite), len(train_pos_edges), np.max(edgelist_bipartite[:, 0]), None)
    for edge in train_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(0)

    test_pos_edges = test_pos_edges.astype(int)
    testing_set = test_pos_edges[:, 1] if testing_nodes[0] <= np.max(edgelist_bipartite[:, 0]) else test_pos_edges[:, 0]
    testing_set = np.unique(testing_set)
    X_test = []
    y_test = []
    for node in testing_set:
        idx = np.where(test_pos_edges[:, 0] == node) if node <= np.max(edgelist_bipartite[:, 0]) else np.where(test_pos_edges[:, 1] == node)
        nodes = test_pos_edges[idx[0], 1] if node <= np.max(edgelist_bipartite[:, 0]) else test_pos_edges[idx[0], 0]
        nodes = nodes[np.in1d(nodes, testing_nodes)]
        if node <= np.max(edgelist_bipartite[:, 0]):
            test_neg_edges = np.append(np.ones((num_v, 1)) * node, np.reshape(np.unique(edgelist_bipartite[:, 1]), (-1, 1)), axis=1)
        else:
            test_neg_edges = np.append(np.reshape(np.unique(edgelist_bipartite[:, 0]), (-1, 1)), np.ones((num_u, 1)) * node, axis=1)
        test_neg_edges = test_neg_edges.astype(int)
        idx = np.random.choice(len(test_neg_edges), int(len(nodes)))
        test_neg_edges = test_neg_edges[idx, :]

        # construct X_test, y_test
        for neighbor in nodes:
            if node <= np.max(edgelist_bipartite[:, 0]):
                node_u_emb = embedding_look_up[node]
                node_v_emb = embedding_look_up[neighbor]
            else:
                node_u_emb = embedding_look_up[neighbor]
                node_v_emb = embedding_look_up[node]
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

    # clf1 = LogisticRegression(random_state=None, solver='lbfgs', max_iter=1000)
    clf1 = XGBClassifier(learning_rate= 0.4, max_depth= 4, min_child_weight=2)
    clf1.fit(X_train, y_train)
    y_pred_proba = clf1.predict_proba(X_test)[:, 1]
    f = open('results.txt', 'a')

    """precision_at_k = []
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        precision_at_k.append(str(precision_at_k_score(y_test, y_pred_proba, i)))
    precision_at_k = ','.join(precision_at_k)
    f.write(f'{precision_at_k};')
    c = list(zip(X_test, y_test))
    random.shuffle(c)
    X_test, y_test = zip(*c)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred_proba = clf1.predict_proba(X_test)[:, 1] """

    auc = roc_auc_score(y_test, y_pred_proba)
    aupr = average_precision_score(y_test, y_pred_proba)
    precision_at_k = precision_at_k_score(y_test, y_pred_proba, 100)
    f.write(f'({precision_at_k:.3f},{auc:.3f}, {aupr:.3f})')
    f.write(';')
    
    return