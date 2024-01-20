from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
import tensorflow.compat.v1 as tf

import time
import os

from utils import construct_feed_dict, sparse_to_tuple
from model import GCNModel
from optimizer import Optimizer

from sklearn.metrics import r2_score


flags = tf.app.flags
FLAGS = flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def r2(pair):
    label = pair[0]
    pred = pair[1]
    return r2_score(np.array(label.todense()), pred)


def train(adj, similarities, features):
    tf.reset_default_graph()

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'similarities1': tf.sparse_placeholder(tf.float32),
        'similarities2': tf.sparse_placeholder(tf.float32),
        'adj_label': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    n = len(similarities[0]) + 1
    num_split = similarities[0][0].shape[0]
    for i in range(2):
        similarities[i] = [x / np.max(x) for x in similarities[i]]
        similarities[i] = sp.vstack(similarities[i])
        similarities[i] = sparse_to_tuple(similarities[i])
    similarities[2] = sparse_to_tuple(similarities[2] / np.max(similarities[2]))

    model = GCNModel(placeholders, num_features,
                     features_nonzero, n, num_split)
    with tf.name_scope('optimizer'):
        opt = Optimizer(model.reconstructions, tf.sparse_tensor_to_dense(placeholders['similarities1'], validate_indices=False), tf.sparse_tensor_to_dense(
            placeholders['similarities2'], validate_indices=False), tf.sparse_tensor_to_dense(placeholders['adj_label'], validate_indices=False), n)

    # Initialize session
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    best_acc = -np.inf

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        feed_dict = construct_feed_dict(
            adj, similarities[0], similarities[1], similarities[2], features, placeholders)
        outs = sess.run([opt.costs, opt.opt_op, opt.r2], feed_dict=feed_dict)

        # save model
        if np.mean(outs[2]) >= np.mean(best_acc):
            saver.save(sess, 'models/' + FLAGS.dataset)
            best_acc = outs[2]

        # print loss and accuracy
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.5e}".format(np.mean(
            outs[0])), "r2=", "{:.5f}".format(np.mean(outs[2])), "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    # restore best model and reconstruct embeddings
    saver.restore(sess, 'models/' + FLAGS.dataset)
    emb = [None for i in range(len(similarities))]
    feed_dict = construct_feed_dict(
        adj, similarities[0], similarities[1], similarities[2], features, placeholders)
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    return np.split(emb, n)
