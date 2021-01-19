import numpy as np
import tensorflow as tf
import pdb
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from .legislator_sentiment_model import LegislatorSentimentModel
epsilon = 1e-30 # global value to prevent probabilities from becoming zero
INF = 1e30

class OrdinalModel(LegislatorSentimentModel):
    # See parent class for constructor
    # Attributes made: input_legs, input_days, labels, num_classes, add_leg_bias
    
    def _map_embs_to_features(self, emb_leg, emb_days):
        """ Maps legislator/day embeddings to ordinal variables (features) """
        map_weights = tf.get_variable("ordinal_map_weights",
                                      shape=[self.poli_dim, self.poli_dim])
        ord_vars = tf.matmul(map_weights, emb_leg,
                             transpose_a=False, transpose_b=True)
        ord_vars = tf.matmul(emb_days, ord_vars,
                             transpose_a=False, transpose_b=False)
        leg_biases = tf.squeeze(tf.gather(self.leg_biases, self.input_legs))
        ordinal_vars = tf.diag_part(ord_vars) # vector of length num leg-day pairs
        if self.add_leg_bias:
            print("\nUsing legislator bias for sentiment\n")
            ordinal_vars = tf.add(ordinal_vars, leg_biases, name="add_leg_bias")
        else:
            print("\nNot using legislator bias for sentiment\n")
        return tf.expand_dims(ordinal_vars, -1)
        
    def _get_logits(self):
        self.ord_thresholds = self._get_ordinal_thresholds()
        self.sigmoid_matrix = self._sigmoid_over_classDiffs()
        return self._all_class_probs()
    
    def _get_ordinal_thresholds(self, init_vals=None):
        if init_vals is not None:
            """ Note that right now, these are constants -- NOT TUNABLE """
            ordinal_thresholds = tf.constant(init_vals, name="ord_thresholds")
        else:
            init_thresholds = np.random.uniform(-5, 5, self.num_classes-1)
            init_thresholds = np.sort(init_thresholds)
            thresholds = []
            """
            temp_threshold = tf.get_variable("neg_inf_thresh", initializer=-INF,
                                             trainable=False)
            thresholds.append(tf.cast(temp_threshold, tf.float64))
            """
            for i in range(self.num_classes-1):
                threshold_name = "threshold_"+str(i)
                #constraint = lambda x: tf.clip_by_value(x-thresholds[i], 0.001, np.infty)
                temp_threshold = tf.get_variable(threshold_name,
                                                 initializer=init_thresholds[i],
                                                 trainable=True)
                thresholds.append(temp_threshold)
            """
            temp_threshold = tf.get_variable("inf_threshold", initializer=INF,
                                             trainable=False)
            thresholds.append(tf.cast(temp_threshold, tf.float64))
            """
            ordinal_thresholds = tf.stack(thresholds)
        return ordinal_thresholds

    def _threshold_ordVar_diff(self):
        """ The difference is accomplished via broadcasting.
        Does C_l - z_it for all thresholds """
        thresholds = tf.cast(self.ord_thresholds, tf.float32)
        ordinal_vars = tf.cast(self.features, tf.float32)
        differences = tf.subtract(thresholds, ordinal_vars)
        return differences

    def _sigmoid_over_classDiffs(self):
        return tf.sigmoid(self._threshold_ordVar_diff())

    def _all_class_probs(self):
        zeros_dim = tf.stack([tf.shape(self.sigmoid_matrix)[0], 1])
        zeros_column = tf.fill(zeros_dim, 0.0)
        ones_dim = tf.stack([tf.shape(self.sigmoid_matrix)[0], 1])
        ones_column = tf.fill(ones_dim, 1.0)
        matrix_1 = tf.concat([zeros_column, self.sigmoid_matrix], axis=1)
        matrix_2 = tf.concat([self.sigmoid_matrix, ones_column], axis=1)
        all_class_probs = tf.subtract(matrix_2, matrix_1)
        return all_class_probs + epsilon
    
    def get_ordinal_loss(self):
        labels_one_hot = tf.one_hot(self.labels, depth=self.num_classes)
        log_probs = tf.log(self.logits)
        masked_log_probs = tf.math.multiply(labels_one_hot, log_probs)
        total_log_prob = tf.reduce_sum(masked_log_probs)
        return -total_log_prob
    
    def total_loss(self, day_embs=None):
        self.features = self._get_features(day_embs)
        self.logits = self._get_logits()
        self.total_loss = self.get_ordinal_loss()
        return self.total_loss
        
