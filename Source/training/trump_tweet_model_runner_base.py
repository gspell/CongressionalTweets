import numpy as np
import tensorflow as tf
import pickle
import os
import argparse
import pdb
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
sys.path.append("../../")

import model_utils as utils
import model_analysis as analysis

class TweetModelRunner():
    def __init__(self, model_specs, model_options, train_options):
        self.model_specs = model_specs
        self.model_options = model_options
        self.train_options = train_options
        
        self.placeholders = self.get_placeholders()
        self.model = self.build_model()
        self.build_update_ops()
        self.loss = self.loss()
        self.optimizer = self.optimizer()
        self.train_op = self.train_op()
        self.sess = self.session()
        
    def get_placeholders(self):
        raise NotImplementedError
    
    def get_feedDict(self):
        raise NotImplementedError
        
    def build_model(self):
        raise NotImplementedError

    def loss(self):
        print("\nMaking loss-op")
        return self.model.get_total_loss()        
    
    def optimizer(self):
        print("Using Adam Optimizer")
        return tf.train.AdamOptimizer(self.train_options["learning_rate"])
    
    def train_op(self):
        return self.optimizer.minimize(self.loss)

    def session(self):
        print("Initializing TensorFlow Session")
        sess = utils.configure_session()
        return sess

    def init_global_vars(self):
        print("Initializing global variables")
        self.sess.run(tf.global_variables_initializer())

    def build_update_ops(self):
        if self.model_options["emb_cstr"] is not None:
            self.emb_constraint_op = self.project_emb_constraint()
        if self.model_options["bias_cstr"] is not None:
            self.bias_constraint_op = self.project_bias_constraint()
        self.trump_embs_update = self.update_trump_embs_op()

    def update_trump_embs_op(self):
        return self.model.update_trump_embs()

    def update_trump_embs(self, feed_dict):
        self.sess.run(self.trump_embs_update, feed_dict=feed_dict)

    def project_emb_constraint(self):
        print("Imposing Embedding norm constraint via projection")
        return self.model.update_leg_embs(self.model.norm_pwr_leg_embs())

    def project_bias_constraint(self):
        print("Imposing Bias norm constraint via projection")
        return self.model.update_leg_biases(self.model.norm_pwr_leg_biases())

    def perform_attribute_updates(self):
        if self.model_options["emb_cstr"] is not None:
            self.sess.run(self.emb_constraint_op)
        if self.model_options["bias_cstr"] is not None:
            self.sess.run(self.bias_constraint_op)
            
    def run_train_step(self):
        raise NotImplementedError
    
    def run_training_loop(self, num_iters, num_eval, train_dataset,
                          eval_feed_dict, trump_tweet_data):
        print("\nCommencing training loop\n")
        self.train_loss = 0
        for i in range(num_iters):
            self.run_train_step(train_dataset, trump_tweet_data)
            if i % num_eval==0:
                self.run_eval(i, eval_feed_dict)
                
    def run_eval(self):
        raise NotImplementedError

    def print_metrics(self):
        raise NotImplementedError
    
    def finish_training(self, train_feed_dict, eval_feed_dict):
        print("\nConcluding training. Updating Trump embeddings\n")
        self.update_trump_embs(train_feed_dict)
        self.update_trump_embs(eval_feed_dict)
        train_metrics= self.get_final_metrics(train_feed_dict)
        eval_metrics = self.get_final_metrics(eval_feed_dict)
        return train_metrics, eval_metrics

    def check_for_exploded_attributes(self):
        attributes = self.get_leg_model_attributes()
        analysis.check_unbounded_attribute(attributes[0], name="leg embs")
        analysis.check_unbounded_attribute(attributes[1], name="trump embs")
        analysis.check_unbounded_attribute(attributes[2], name="leg biases")
        
    def check_satisfied_constraints(self):
        if self.model_options["emb_cstr"]=="normalize":
            print("Checking legislator embedding normalization satisfied")
            leg_embs = self.model.leg_embs
            analysis.check_attribute_normalization(self.sess.run(leg_embs))
        if self.model_options["bias_cstr"]=="normalize":
            print("Checking legislator bias normalization satisfied")
            leg_biases = self.model.leg_biases
            analysis.check_attribute_normalization(self.sess.run(leg_biases))
    
    def get_final_metrics(self, feed_dict):
        raise NotImplementedError
    
    def get_leg_model_attributes(self):
        output_list = [self.model.leg_embs, self.model.trump_embs,
                       self.model.leg_biases]
        leg_model_attributes = self.sess.run(output_list)
        return leg_model_attributes

    def get_text_model_attributes(self):
        """
        output_list = [self.model.word_embs, self.model.word_map]
        text_model_attributes = self.sess.run(output_list)
        """
        if self.model_options["use_text"] is False:
            return None
        text_model_attribute_op = self.model.get_all_text_model_attributes()
        text_model_attributes = self.sess.run(text_model_attribute_op)
        return text_model_attributes
