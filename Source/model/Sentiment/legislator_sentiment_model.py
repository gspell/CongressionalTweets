import numpy as np
import tensorflow as tf
import sys
sys.path.append("../../")
from model.legislator_model import LegislatorModel

epsilon=1e-30 # global value to prevent probabilities from becoming zero
INF = 1e30

class LegislatorSentimentModel(LegislatorModel):
    def __init__(self, model_specs, add_leg_bias, input_legs, input_days,
                 labels, emb_constraint=None):
        """model_specs is dict with poli_dim, num_legis, num_days, num_classes"""
        # Inherit from parent class: trump_embs, leg_embs, leg_biases
        print("--Creating legislator tweet-sentiment model")
        super().__init__(model_specs, emb_constraint)
        self.input_legs = input_legs
        self.input_days = input_days
        self.labels = labels
        self.num_classes = model_specs["num_classes"]
        self.add_leg_bias = add_leg_bias

    def _get_features(self, day_embs=None):
        emb_days = self._get_day_embs(day_embs)
        emb_leg = tf.nn.embedding_lookup(self.leg_embs, self.input_legs)
        features = self._map_embs_to_features(emb_leg, emb_days)
        return features

    def _get_day_embs(self, day_embs=None):
        if day_embs is None:
            emb_days = tf.nn.embedding_lookup(self.trump_embs, self.input_days)
        else:
            emb_days = day_embs
        return emb_days
    
    def _map_embs_to_features(self, embedded_legis, embedded_days):
        raise NotImplementedError
    
    def _get_logits(self):
        raise NotImplementedError

    def _predicted_labels(self):
        predicted_labels = tf.argmax(self.logits, axis=1, name="pred_labels")
        return tf.cast(predicted_labels, tf.int32)
    
    def get_error_metric(self, metric="MAE"):
        errors = tf.subtract(self.labels, self._predicted_labels())
        errors = tf.cast(errors, tf.float32)
        if metric == "MAE":
            metric = tf.abs(errors)
        elif metric == "MSE":
            metric = tf.square(errors)
        return tf.reduce_mean(metric)

    def get_accuracy(self):
        correct_predictions = tf.equal(self._predicted_labels(), self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        return accuracy
