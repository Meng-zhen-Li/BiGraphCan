import tensorflow.compat.v1 as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

devices = tf.config.get_visible_devices('GPU')
if len(devices) == 0:
    devices = tf.config.get_visible_devices()
devices = [device.name.replace('physical_device:', '') for device in devices]

class Optimizer(object):
    def __init__(self, preds, labels1, labels2, labels3, num_graphs):
        preds[:2] = [tf.split(x, num_graphs-1) for x in preds[:2]]
        labels1 = tf.split(labels1, num_graphs-1)
        labels2 = tf.split(labels2, num_graphs-1)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.costs = []
        self.r2 = []
        self.grads_vars = None

        # Compute loss
        for sim_idx in range(num_graphs - 1):
            with tf.device(devices[sim_idx % len(devices)]):
                unexplained_error = tf.reduce_sum(tf.square(labels1[sim_idx] - preds[0][sim_idx])) + tf.reduce_sum(tf.square(labels2[sim_idx] - preds[1][sim_idx]))
                total_error = tf.reduce_sum(tf.square(labels1[sim_idx] - tf.reduce_mean(labels1[sim_idx], axis=0))) + tf.reduce_sum(tf.square(labels2[sim_idx] - tf.reduce_mean(labels2[sim_idx], axis=0)))
                R2 = 1. - tf.div(unexplained_error, total_error)
                self.r2.append(R2)
        unexplained_error = tf.reduce_sum(tf.square(labels3 - preds[2]))
        total_error = tf.reduce_sum(tf.square(labels3 - tf.reduce_mean(labels3, axis=0)))
        R2 = 1. - tf.div(unexplained_error, total_error)
        self.r2.append(R2)

        for sim_idx in range(num_graphs - 1):
            with tf.device(devices[sim_idx % len(devices)]):
                cost = tf.losses.mean_squared_error(labels=labels1[sim_idx], predictions=preds[0][sim_idx]) + tf.losses.mean_squared_error(labels=labels2[sim_idx], predictions=preds[1][sim_idx])
                grads_var = self.optimizer.compute_gradients(cost)
                self.costs.append(cost)
                if self.grads_vars is None:
                    self.grads_vars = [(x[0], x[1]) for x in grads_var[:2]]
                    self.grads_vars.append(grads_var[2])
                else:
                    self.grads_vars[:2] = [(x[0] + y[0], x[1]) for x, y in zip(self.grads_vars[:2], grads_var[:2])]
                    self.grads_vars[2] = (self.grads_vars[2][0] + grads_var[2][0], self.grads_vars[2][1])

        cost = tf.losses.mean_squared_error(labels=labels3, predictions=preds[-1])
        grads_var = self.optimizer.compute_gradients(cost)
        self.costs.append(cost)
        self.grads_vars[:2] = [(x[0] + y[0], x[1]) for x, y in zip(self.grads_vars[:2], grads_var[:2])]
        self.grads_vars[2] = (self.grads_vars[2][0] + grads_var[2][0], self.grads_vars[2][1])
        # Apply Gradients
        self.opt_op = self.optimizer.apply_gradients(self.grads_vars)