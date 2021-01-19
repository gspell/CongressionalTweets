import numpy as np
import tensorflow as tf
import pickle
import os
import sys
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
sys.path.append("../")

""" Utilities that I've found useful and are repeated across the various model
modules being constructed """

def configure_session():
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    return tf.Session(config=session_config)
    
def set_random_seeds(seed=0):
    tf.set_random_seed(seed=seed)
    np.random.seed(seed=seed)

def get_countQuantities(count_matrix):
    num_legislators, num_days = np.shape(count_matrix)
    nums_tweets = np.sum(count_matrix, axis=1)
    return num_legislators, num_days, nums_tweets

""" PLACEHOLDERS UTILS """
def get_count_placeholders(num_legislators, num_days):
    counts_shape = [num_legislators, num_days]
    counts = tf.placeholder(dtype=tf.float32, shape=counts_shape, name="counts")
    count_days = tf.placeholder(dtype=tf.int32, shape=[None], name="counts_days")
    count_legs = tf.placeholder(dtype=tf.int32, shape=[None], name="counts_legs")
    return counts, count_days, count_legs

def get_text_placeholders(seq_length=200, num_inputs=None, name="in_text"):
    text_shape = [num_inputs, seq_length] 
    input_text = tf.placeholder(dtype=tf.int32, shape=text_shape, name=name)
    return input_text

def get_sentiment_placeholders(num_inputs=None):
    shape=[num_inputs]
    labeled_legs = tf.placeholder(dtype=tf.int32, shape=shape, name="lbld_legs")
    labeled_days = tf.placeholder(dtype=tf.int32, shape=shape, name="lbld_days")
    labels = tf.placeholder(dtype=tf.int32, shape=shape, name="labels")
    return labeled_legs, labeled_days, labels

""" FEED_DICT UTILS """
def get_sentiment_feedDicts(placeholders, train_data, eval_data):
    """ This method is not recommended for use right now """
    input_legs, input_days, labels = placeholders
    train_feed_dict = {input_legs: train_data[0], input_days: train_data[1],
                       labels: train_data[2]}
    eval_feed_dict = {input_legs: val_data[0], input_days: eval_data[1],
                     labels: val_data[2] }
    return train_feed_dict, val_feed_dict

""" INITIALIZATION UTILS """
def init_leg_biases(sess, model, train_unique_legs, num_legis):
    print("\nInitializing legislator biases based on party")
    init_vals = np.expand_dims(get_init_leg_biases(), axis=1)
    """
    for leg in range(num_legis):
        if leg not in train_unique_legs:
            sess.run(model.leg_biases[leg].assign(init_vals[leg]))
    """
    sess.run(model.leg_biases.assign(init_vals))
    return sess

def get_init_leg_biases(leg_dict=None):
    if leg_dict is None:
        leg_dict = load_legislator_dict("../data/legislator_dict.pickle")
    init_vals = []
    for key, val in leg_dict.items():
        if val["party"]=="R":
            init_vals.append(5)
        else:
            init_vals.append(-10)
    return init_vals

def load_word_embeddings(emb_filename):
    with open(emb_filename, "rb") as pickle_file:
        word_embs = pickle.load(pickle_file)
    return word_embs

def init_word_embeddings(emb_filename, sess, model):
    if emb_filename is not None:
        print("\nInitializing word embeddings from {}".format(emb_filename))
        init_embs = load_word_embeddings(emb_filename)
        sess.run(model.SWEM_model.source_embedding.assign(init_embeddings))
    else:
        print("Using randomly initialized word embeddings")
    return sess

""" CONSTRAINT UTILS """
def l2_normalize(x, axis=None, epsilon=1e-12, name=None):
    """Normalize along dimension `axis` using an L2 norm.

    For a 1-D tensor with `axis=0` compus
        output = x / sqrt(max(sum(x**2)), epsilon))
    """
    with ops.name_scope(name, "l2_normalize", [x]) as name:
        x_squared = math_ops.square(ops.convert_to_tensor(x, name="x"))
        square_sum = math_ops.reduce_sum(x_squared, axis, keep_dims=True)
        x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
        return math_ops.multiply(x, x_inv_norm, name=name)
