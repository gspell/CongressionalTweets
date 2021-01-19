import numpy as np
import tensorflow as tf
import os
import pdb
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.ERROR)
import sys
sys.path.append("../")

from .legislator_model import LegislatorModel
from .Counts.legislator_count_model import LegislatorCountModel
from .Sentiment.ordinal_model import OrdinalModel
from .Text.trump_text_model import TrumpTextModel

class TrumpTweetModel(LegislatorModel):
    def __init__(self, model_specs, model_options, labeled_legs, labeled_days,
                 labels, in_text, count_legs, count_text, count_days):
        """ This now accepts dicts to describe the model. Those dicts are:
            a) model_specs: dict with poli_dim, num_legis, num_days, num_classes,
                            loss_ratio, text_params (formerly emb_params)
            b) model_options: another dict (separate from model_specs) with
                              add_leg_bias, use_text, emb_constraint, 
                              bias_contraint, use_nonlinearity
        Things still to possibly change:
            1) Have the placeholders passed in differently, possibly???
            Possible data structures to use:
              a) placeholders_dict: maybe want to put this elsewhere than init??
        """
        print("\nCreating joint legislator sentiment/count model")
        #emb_constraint = model_options["emb_cstr"]
        super().__init__(model_specs, emb_constraint=None)
        self.count_days = tf.cast(count_days, tf.int32) # a placeholder
        self.count_legs = tf.cast(count_legs, tf.int32) # a placeholder
        self.count_text = count_text # a placeholder
        self.in_text = in_text # a placeholder
        self.loss_ratio = model_specs["loss_ratio"]
        self.use_text = model_options["use_text"] # default: True
        if self.use_text:
            print("--Creating Trump tweet text model")
            self.text_model = TrumpTextModel(model_specs,
                                             model_options["use_nonlinearity"])
            self._check_attribute_equivalence()
        self.count_model = LegislatorCountModel(model_specs)
        self.sentiment_model = OrdinalModel(model_specs, model_options["add_leg_bias"],
                                            labeled_legs, labeled_days, labels)
        self.word_embs = self.get_word_embs()
        self.word_map = self.get_word_map()
        
    def get_error_metric(self):
        return self.sentiment_model.get_error_metric()

    def get_accuracy(self):
        return self.sentiment_model.get_accuracy()
    
    def get_sentiment_loss(self, day_embs):
        return self.sentiment_model.total_loss(day_embs)
    
    def get_count_loss(self, counts, day_embs, nums_tweets=None):
        self.count_loss = self.count_model.get_loss(
            counts, self.count_days, day_embs, self.count_legs, nums_tweets)        
        return self.count_loss
               
    def get_word_embs(self):
        if self.use_text is True:
            return self.text_model.SWEM_model.word_embs
        else:
            return None
        
    def get_word_map(self):
        if self.use_text is True:
            return self.text_model.emb_map_weights
        else:
            return None
        
    def get_all_text_model_attributes(self):
        if self.use_text is True:
            return self.text_model.get_all_text_model_attributes()
        else:
            return None
        
    def get_day_embs(self, text, day_idxs):
        if self.use_text:
            embs = self.text_model.construct_trump_embs(text, day_idxs)
            return embs
        else:
            return tf.nn.embedding_lookup(self.trump_embs, day_idxs)

    def update_trump_embs(self):
        if self.use_text:
            embs = self.text_model.map_text_to_trump_embeddings(self.count_text)
            all_embs = tf.scatter_update(self.trump_embs, self.count_days, embs)
            self._check_attribute_equivalence()
            return all_embs
        else:
            return self.trump_embs
    
    def get_total_loss(self, counts, nums_tweets=None):        
        count_day_embs = self.get_day_embs(self.count_text, self.count_days)
        sent_days = self.sentiment_model.input_days
        sent_day_embs = self.get_day_embs(self.in_text, sent_days)

        count_loss = self.get_count_loss(counts, count_day_embs, nums_tweets)
        self.sentiment_loss = self.get_sentiment_loss(sent_day_embs)
        scaled_count_loss = self.loss_ratio * count_loss
        scaled_sent_loss = (1 - self.loss_ratio) * self.sentiment_loss
        self.total_loss = scaled_count_loss + scaled_sent_loss
        return self.total_loss

    def _check_attribute_equivalence(self):
        assert self.trump_embs is self.text_model.trump_embs
        assert self.day_biases is self.text_model.day_biases
