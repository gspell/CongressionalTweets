import numpy as np
import tensorflow as tf

import sys
sys.path.append("../")
epsilon = 1e-30 # global value to be added to keep values (probabilities) from being zero

def compute_distn_params(leg_embs, trump_embs, leg_biases=None, day_biases=None):
    params = map_attributes_to_params(leg_embs, trump_embs, leg_biases, day_biases)
    params = impose_param_nonnegativity(params, method="exponentiate")
    return params

def impose_param_nonnegativity(params, method="exponentiate"):
    """ Note that the parameter of a poisson distribution must be > 0"""
    if method=="abs":
        params = tf.abs(params) + epsilon
    elif method=="exponentiate":
        """ THIS SHOULD BE THE ONE"""
        params = tf.exp(params) + epsilon
    elif method=="relu":
        params = tf.nn.relu(params) + epsilon
    elif method=="normalize":
        """ Kind of a misnaming: really just a DC-bias add"""
        params = tf.add(params, tf.reduce_min(params)) + epsilon
    elif method=="softmax":
        params = tf.nn.softmax(params) + epsilon
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

def construct_distns(params, num_tweets=None):
    params = tf.cast(params, dtype=tf.float32)
    distns = tf.contrib.distributions.Poisson(rate=params) # consider using log_rate!
    return distns

def counts_logProb(counts, distn):
    counts = tf.cast(counts, tf.float32)
    log_prob = distn.log_prob(counts)
    return tf.cast(log_prob, tf.float32)
