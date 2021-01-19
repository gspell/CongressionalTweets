import numpy as np
import tensorflow as tf
import pickle
import os
import argparse
import pdb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.append("../../")

import model.Counts.poisson_functions as poisson_fcns
import model.Counts.negative_binomial_fcns as neg_bin_fcns
from model.legislator_model import LegislatorModel

class LegislatorCountModel(LegislatorModel):
    def __init__(self, model_specs, emb_constraint=None,
                 add_leg_bias=False, add_day_bias=False):
        """model_specs is dict including poli_dim, num_legis, num_days
           model_specs also now has distn, which specifies poisson or neg_bin"""
        print("--Creating legislator tweet-count model")
        super().__init__(model_specs, emb_constraint)
        self.count_distn_fcns = self.set_count_distn(model_specs)
        self.add_leg_bias, self.add_day_bias = add_leg_bias, add_day_bias

    def set_count_distn(self, model_specs):
        if model_specs["distn"]=="poisson":
            print("--Using a Poisson count distribution")
            count_distn_fcns = poisson_fcns
            self.neg_bin_param = None
        elif model_specs["distn"]=="negative_binomial":
            print("--Using a Negative Binomial count distribution")
            count_distn_fcns = neg_bin_fcns
            self.neg_bin_param = model_specs["neg_bin_param"]
        else:
            NotImplementedError("Can only do poisson or negative_binomial")
        return count_distn_fcns
    
    def get_distn_params(self, leg_embs, day_embs, days_idxs):
        leg_biases, day_biases = self.set_biases(days_idxs)
        params = self.count_distn_fcns.compute_distn_params(leg_embs, day_embs,
                                                        leg_biases, day_biases)
        return params

    def set_biases(self, day_idxs):
        leg_biases = self.leg_biases if self.add_leg_bias else None
        if self.add_day_bias:
            day_biases = tf.gather(self.day_biases, day_idxs)
        else:
            day_biases = None
        return leg_biases, day_biases

    def index_count_matrix(self, counts, day_idxs, leg_idxs):
        counts = tf.gather(counts, day_idxs, axis=1)
        if leg_idxs is not None:
            counts = tf.gather(counts, leg_idxs, axis=0)
        return counts
    
    def get_NegLogLike(self, counts, num_tweets, distn_params):
        if self.neg_bin_param is not None:
            distns = self.count_distn_fcns.construct_distns(distn_params,
                                                            self.neg_bin_param,
                                                            num_tweets)
        else:
            distns = self.count_distn_fcns.construct_distns(distn_params,num_tweets)
        counts_logLikes = self.count_distn_fcns.counts_logProb(counts, distns)
        return -tf.reduce_sum(counts_logLikes)

    def get_leg_embs(self, leg_idxs=None):
        if leg_idxs is None:
            #Use all legislators, unless otherwise specified
            leg_embs = self.leg_embs
        else:
            leg_embs = tf.nn.embedding_lookup(self.leg_embs, leg_idxs)
        return leg_embs
    
    def get_loss(self, counts, day_idxs, day_embs, leg_idxs=None, nums_tweets=None):
        leg_embs = self.get_leg_embs(leg_idxs)
        self.distn_params = self.get_distn_params(leg_embs, day_embs, day_idxs)
        counts = self.index_count_matrix(counts, day_idxs, leg_idxs)
        NLL = self.get_NegLogLike(counts, nums_tweets, self.distn_params)
        return NLL
