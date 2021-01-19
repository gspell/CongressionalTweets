from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import functools

class SWEM:  
  def __init__(self, network_architecture, drop_keep=1.0): 
    self.drop_keep = drop_keep
    print("----Creating SWEM")
    self._initialize_network_params(network_architecture)
    self.word_embs = self.init_word_embs()

  def _initialize_network_params(self, network_architecture):
    self.seq_length = network_architecture['text_length']
    self.vocab_size = network_architecture['vocab_size']
    self.word_emb_size = network_architecture['word_emb_size']
  
  def init_word_embs(self):
    shape=[self.vocab_size, self.word_emb_size]
    print("----Randomly initializing word embeddings")
    initializer = tf.random_uniform(shape, -1.0, 1.0, seed=0)
    word_embs = tf.Variable(initializer, trainable=True, name="W_emb")
    #word_embs = tf.get_variable("word_embs", initializer=initializer)
    #word_embs = tf.get_variable("word_embs", shape=shape)
    return word_embs
  
  def embed_inputs(self, input_idxs):
    embedded_inputs = tf.nn.embedding_lookup(self.word_embs, input_idxs)
    #normalized_word_embs = tf.math.l2_normalize(embedded_inputs, axis=1)
    return embedded_inputs
  
  def get_max_pooled_feature_vector(self, input_seqs):
    return tf.reduce_max(self.embed_inputs(input_seqs), axis=1)
        
