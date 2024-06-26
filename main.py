from __future__ import division
from __future__ import print_function

from evaluation import LinkPrediction, ZeroShot

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
flags.DEFINE_string('testing_nodes', '0', 'The node to test.')
flags.DEFINE_string('task', 'cold_start', 'cold_start or zero_shot')
flags.DEFINE_float('noise', 0, 'Add noise to the graphs')

dataset_str = FLAGS.dataset
adj, num_split, features, adj_train, edgelist_bipartite = load_data(dataset_str)

features = preprocess_features(features)

# Training and Evaluation
adj_norm = preprocess_graph(adj_train)
similarities = similarity_matrix(adj_train, num_split)

emb = train(adj_norm, similarities, features)
emb = consensus(emb)
if FLAGS.task == 'cold_start':
    LinkPrediction(emb, edgelist_bipartite, [int(x) for x in FLAGS.testing_nodes.split(',')])
else:
    ZeroShot(emb, edgelist_bipartite, [int(x) for x in FLAGS.testing_nodes.split(',')])