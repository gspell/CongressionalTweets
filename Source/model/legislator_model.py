import numpy as np
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

import pdb

class LegislatorModel():
    def __init__(self, model_specs, emb_constraint=None):
        """ 
        model_specs is a dict that must include poli_dim, num_legis, num_days
        """
        self.poli_dim = model_specs["poli_dim"]
        self.num_legis = model_specs["num_legis"]
        self.num_days = model_specs["num_days"]

        #self.emb_constraint = model_options["emb_constraint"]
        self.emb_constraint = emb_constraint
        self._create_model_attributes()
        
    def _create_model_attributes(self):
        with tf.variable_scope("attributes", reuse=tf.AUTO_REUSE) as scope:
            print("--Creating legislator tweet model")
            if self.emb_constraint is not None:
                msg = "----Using emb constraint: {}".format(self.emb_constraint)
                print(msg)
                print("----Note: emb constraint this way NOT recommended!!!!!")
            self.trump_embs = self._get_trump_embeddings()
            self.leg_embs = self._get_legislator_embeddings()
            self.leg_biases = self._get_legislator_biases()
            self.day_biases = self._get_day_biases()
            
    def _get_trump_embeddings(self):
        shape = [self.num_days, self.poli_dim]
        if self.emb_constraint=="Nonnegative":
            trump_embeddings = self._nonnegative_variable(shape, "trump_embs")
        elif self.emb_constraint=="normalize":
            trump_embeddings = self._normalized_variable(shape, "trump_embs")
        else:
            trump_embeddings = tf.get_variable("trump_embs", shape=shape)
        return trump_embeddings

    def _get_legislator_embeddings(self):
        shape = [self.num_legis, self.poli_dim]
        if self.emb_constraint=="Nonnegative":
            legislator_embeddings = self._nonnegative_variable(shape, "leg_embs")
        elif self.emb_constraint=="normalize":
            legislator_embeddings = self._normalized_variable(shape, "leg_embs")
        else:
            legislator_embeddings = tf.get_variable("leg_embs", shape=shape)
        return legislator_embeddings

    def _get_legislator_biases(self):
        shape = [self.num_legis, 1]
        if self.emb_constraint=="Nonnegative":
            legislator_biases = self._nonnegative_variable(shape, "leg_biases")
        else:
            legislator_biases = tf.get_variable("leg_biases", shape=shape)
        return legislator_biases

    def _get_day_biases(self):
        shape=[self.num_days, 1]
        if self.emb_constraint=="Nonnegative":
            day_biases = self._nonnegative_variable(shape, "day_biases")
        else:
            day_biases = tf.get_variable("day_biases", shape=shape)
        return day_biases

    def norm_pwr_leg_embs(self):
        embs_shape = [self.num_legis, self.poli_dim]
        normed_embs = self.norm_pwr_projection(self.leg_embs, embs_shape)
        return normed_embs

    def update_leg_embs(self, new_embs):
        return self.leg_embs.assign(new_embs)

    def norm_pwr_leg_biases(self):
        biases_shape = [self.num_legis, 1]
        normed_biases = self.norm_pwr_projection(self.leg_biases, biases_shape)
        return normed_biases

    def update_leg_biases(self, new_biases):
        return self.leg_biases.assign(new_biases)
    
    @staticmethod
    def norm_pwr_projection(embs, embs_shape):
        flat_embs = tf.reshape(embs, [-1])
        normed_embs = tf.reshape(l2_normalize(flat_embs), embs_shape)
        return normed_embs
    
    @staticmethod
    def _nonnegative_variable(shape, name, initializer=None):
        """ See note for normalized_variables """
        if initializer is None:
            initializer = tf.random_uniform(maxval=2, shape=shape)
            #constraint_fcn = lambda x: tf.clip_by_value(x, 0, np.infty)
            return tf.get_variable(name, initializer=initializer)
        else:
            return tf.get_variable(name, shape=shape, initializer=initializer)

    @staticmethod
    def _normalized_variable(shape, name):
        """ Note: using the constraint function DOES NOT WORK when using 
        tf.nn.embedding_lookup. Gives error "cannot use constraint function
        on sparse variable." Because of this, I think it may be best to perform
        the normalization projection explicitly before using the embeddings """  
        constraint = lambda x: tf.reshape(l2_normalize(tf.reshape(x, [-1])),
                                          shape)
        return tf.get_variable(name, shape=shape, constraint=constraint)

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
