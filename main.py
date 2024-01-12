from __future__ import division
from __future__ import print_function

from evaluation import LinkPrediction

from utils import preprocess_graph, preprocess_features
from similarity import similarity_matrix
from input_data import load_data
from train import train
from consensus import consensus

import scipy.sparse as sp
import numpy as np
import networkx as nx
import tensorflow.compat.v1 as tf

import os

tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('dataset', 'DGI', 'Dataset string.')
flags.DEFINE_string('sim_idx', '0', 'To use all similarities(0) or to use one of them(index of similarity).')

dataset_str = FLAGS.dataset
adj, num_split, features, adj_no_bipartite, edgelist_bipartite, testing_nodes = load_data(dataset_str)

features = preprocess_features(features)

# Training and Evaluation
adj_norm = preprocess_graph(adj_no_bipartite)
similarities = similarity_matrix(adj_no_bipartite, num_split)

emb = train(adj_norm, similarities, features)
if len(emb) > 1:
    emb = consensus(emb)
else:
    emb = emb[0]
LinkPrediction(emb, edgelist_bipartite, testing_nodes)