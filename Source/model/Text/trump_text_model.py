import numpy as np
import tensorflow as tf
import pdb

from .SWEM import SWEM


class TrumpTextModel():
    def __init__(self, model_specs, nonlinearity=False):
        self.poli_dim = model_specs["poli_dim"]
        self.num_days = model_specs["num_days"]
        self.text_params = model_specs["text_params"] # is a dictionary
        self.nonlinearity=nonlinearity
        self.trump_embs, self.day_biases = self._initialize_trump_attributes()
        self._make_SWEM_model()
        self._create_map_parameters()
        
    def _initialize_trump_attributes(self):
        print("----Making text model attributes: Trump Embeddings and Biases")
        with tf.variable_scope("attributes", reuse=tf.AUTO_REUSE) as scope:
            emb_shape = [self.num_days, self.poli_dim]
            self.trump_embs = tf.get_variable("trump_embs", shape=emb_shape)
            bias_shape = [self.num_days, 1]
            self.day_biases = tf.get_variable("day_biases", shape=bias_shape)
        return self.trump_embs, self.day_biases

    def _make_SWEM_model(self):
        self.SWEM_model = SWEM(self.text_params)
        
    def _create_map_parameters(self):
        print("----Creating text to Trump embedding map parameters")
        with tf.variable_scope("text_to_emb_map", reuse=tf.AUTO_REUSE) as scope:
            shape_1 = [self.text_params["word_emb_size"], self.poli_dim]
            self.emb_map_weights = tf.get_variable("map_weights", shape_1)

            shape_2 = [self.poli_dim]
            self.emb_map_biases = tf.get_variable("map_biases", shape_2)

            shape_3 = [self.text_params["word_emb_size"], 1]
            self.map_weights_2 = tf.get_variable("bias_map_weights", shape_3)

            shape_4 = [1]
            self.map_biases_2 = tf.get_variable("bias_map_bias", shape_4)
            
        if self.nonlinearity:
            print("----Using a nonlinearity in Trump embedding mapping")
            emb_weights_shape = [self.poli_dim, self.poli_dim]
            self.emb_map_weights_2 = tf.get_variable("trump_emb_map_weights_2",
                                                     shape=emb_weights_shape)
            emb_bias_shape = [self.poli_dim]
            self.emb_map_biases_2 = tf.get_variable("trump_emb_map_biases_2",
                                                    shape=emb_bias_shape)
    
    def map_text_to_trump_embeddings(self, input_text):
        features = self.SWEM_model.get_max_pooled_feature_vector(input_text)
        new_trump_embs = tf.nn.xw_plus_b(features, self.emb_map_weights,
                                         self.emb_map_biases)
        if self.nonlinearity:
            rectified_trump_embs = tf.nn.relu(new_trump_embs)
            mapped_trump_embs = tf.nn.xw_plus_b(rectified_trump_embs,
                                                self.emb_map_weights_2,
                                                self.emb_map_biases_2)
        else:
            mapped_trump_embs = new_trump_embs
            
        return mapped_trump_embs

    def map_text_to_day_biases(self, input_text):
        features = self.SWEM_model.get_max_pooled_feature_vector(input_text)
        mapped_day_biases = tf.nn.xw_plus_b(features, self.map_weights_2,
                                            self.map_biases_2)
        mapped_day_biases = mapped_day_biases
        return mapped_day_biases
    
    def construct_trump_embs(self, in_text, in_days):        
        mapped_embs = self.map_text_to_trump_embeddings(in_text)
        new_embs = tf.scatter_update(self.trump_embs, in_days, mapped_embs)
        with tf.control_dependencies([new_embs]):
            embs_return = tf.identity(mapped_embs)
        return embs_return

    def construct_day_biases(self, in_text, in_days):
        mapped_biases = self.map_text_to_day_biases(in_text)
        new_biases = tf.scatter_update(self.day_biases, in_days, mapped_biases)
        with tf.control_dependences([new_biases]):
            biases_return = tf.identity(mapped_biases)
        return biases_return

    def get_all_text_model_attributes(self):
        """ Package all parameters -- word embs, map weights/biases -- into a 
            dict. Need to handle cases of linear model or nonlinear model """
        text_attribute_dict = {}
        text_attribute_dict["word_embs"] = self.SWEM_model.word_embs
        # This is for the first layer to map to embeddings (group together?)
        text_attribute_dict["emb_map_weights_1"] = self.emb_map_weights
        text_attribute_dict["emb_map_biases_1"] = self.emb_map_biases
        # This is for the second layer to map to embeddings (group together?)
        if self.nonlinearity:
            text_attribute_dict["emb_map_weights_2"] = self.emb_map_weights_2
            text_attribute_dict["emb_map_biases_2"] = self.emb_map_biases_2
        return text_attribute_dict
