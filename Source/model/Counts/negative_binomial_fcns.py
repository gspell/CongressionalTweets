import numpy as np
import tensorflow as tf
import pdb
#import tensorflow_probability as tfp

import sys
sys.path.append("../")
epsilon = 1e-30 # global value to be added to keep values (probabilities) from being zero

def compute_distn_params(leg_embs, trump_embs,
                         leg_biases=None, day_biases=None):
    params = map_attributes_to_params(leg_embs, trump_embs,
                                      leg_biases, day_biases)
    #params = impose_param_nonnegativity(params, method="sigmoid")
    return params

def impose_param_nonnegativity(params, method="normalize"):
    """ Note that the parameter of a poisson distribution must be > 0"""
    if method=="abs":
        params = tf.abs(params) + epsilon
    elif method=="relu":
        params = tf.nn.relu(params) + epsilon
    elif method=="normalize":
        """ Kind of a misnaming: really just a DC-bias add"""
        params = tf.add(params, tf.reduce_min(params)) + epsilon
    elif method=="sigmoid":
        """ This is proper with the negative binomial if using probs arg"""
        params = tf.nn.sigmoid(params) + epsilon
    else:
        params = params + epsilon
    return params

def map_attributes_to_params(leg_embs, trump_embs, leg_biases=None, day_biases=None):
    params = tf.matmul(leg_embs, trump_embs, transpose_a=False, transpose_b=True)
    if leg_biases is not None:
        print("\nAdding legislator bias for count model")
        params = tf.add(params, leg_biases)
    if day_biases is not None:
        print("\nAdding day bias for count model")
        params = tf.add(params, tf.transpose(day_biases))
    return params

def construct_distns(params, neg_bin_param = 1.0, num_tweets=None):
    params = tf.cast(params, dtype=tf.float32)
    non_neg_constraint = lambda x: tf.clip_by_value(x, neg_bin_param, np.infty)
    neg_bin = tf.contrib.distributions.NegativeBinomial
    
    neg_bin_count_param = tf.get_variable("neg_bin_count_param", shape=[1],
                                          dtype=tf.float32,
                                          constraint=non_neg_constraint)
    
    #neg_bin_count_param = neg_bin_param
    distns = neg_bin(total_count = neg_bin_count_param, logits=params)
    #distns = neg_bin(total_count = neg_bin_count_param, probs=params)
    return distns

def counts_logProb(counts, distn):
    counts = tf.cast(counts, tf.float32)
    log_prob = distn.log_prob(counts)
    return tf.cast(log_prob, tf.float32)
