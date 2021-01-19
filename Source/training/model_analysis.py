import numpy as np
import tensorflow as tf
import pickle
import os
import pdb

import sys
sys.path.append("../")

import model_utils as utils
from data_processing.data_loading import load_tweet_count_matrix, \
                                         load_legislator_dict

def get_incorrect_prediction_idx(feed_dict, sess, model, placeholders):
    predictions = sess.run(model.sentiment_model._predicted_labels(),
                           feed_dict=feed_dict)
    incorrect_mask = (predictions!=feed_dict[placeholders[-2]])
    incorrect_idxs = np.squeeze(np.where(incorrect_mask))
    return incorrect_idxs

def get_incorrect_leg_idxs(feed_dict, sess, model, placeholders, leg_dict=None):
    if leg_dict is None:
        leg_dict = load_legislator_dict("../data/legislator_dict.pickle")
    incorrect_prediction_idxs = get_incorrect_prediction_idx(feed_dict, sess,
                                                             model, placeholders)
    incorrect_leg_idxs = feed_dict[placeholders[2]][incorrect_prediction_idxs]
    return incorrect_leg_idxs

def get_list_incorrect_legislators(feed_dict, leg_dict, incorrect_idxs,
                                   placeholders):
    incorrect_leg_idxs = feed_dict[placeholders[2]][incorrect_idxs]
    incorrect_republicans, incorrect_democrats = [], []
    for idx in incorrect_leg_idxs:
        leg = leg_dict[int(idx)]
        if leg["party"]=="R":
            incorrect_republicans.append(leg)
        else:
            incorrect_democrats.append(leg)
    return incorrect_republicans, incorrect_democrats

def get_incorrect_class_idxs(incorrect_idxs, feed_dict, placeholders):
    true_labels = feed_dict[placeholders[-2]]
    incorrect = true_labels[incorrect_idxs]
    idx_neutrals = incorrect_idxs[np.where(incorrect==1)[0]]
    idx_positives = incorrect_idxs[np.where(incorrect==2)[0]]
    idx_negatives = incorrect_idxs[np.where(incorrect==0)[0]]
    return idx_neutrals, idx_positives, idx_negatives

def get_party_class_counts(leg_idxs, leg_dict, labels):
    # Note data is a tuple of form (leg_idx, day_idx, label)
    party_keys = ['Republicans', 'Democrats']
    class_keys = ['Negatives', 'Neutrals', 'Positives']
    # Initialize count dictionary using dict-comprehension
    party_class_counts = {key: {key_2: 0 for key_2 in class_keys} for
                          key in party_keys}
    for i in range(len(leg_idxs)):
        leg_idx, label = leg_idxs[i], labels[i]
        leg = leg_dict[int(leg_idx)]
        if leg["party"]=="R":
            if label==0:
                party_class_counts['Republicans']['Negatives'] += 1
            elif label==1:
                party_class_counts['Republicans']['Neutrals'] += 1
            elif label==2:
                party_class_counts['Republicans']['Positives'] += 1
            else:
                raise ValueError('Label is not 0, 1, or 2')
        else:
            if label==0:
                party_class_counts['Democrats']['Negatives'] += 1
            elif label==1:
                party_class_counts['Democrats']['Neutrals'] +=1
            elif label==2:
                party_class_counts['Democrats']['Positives'] +=1
            else:
                raise ValueError("Label is not 0, 1, or 2")
    return party_class_counts

def dataset_incorrect_breakdown(feed_dict, sess, model, placeholders,
                                leg_dict=None):
    if leg_dict is None:
        leg_dict = load_legislator_dict("../data/legislator_dict.pickle")
    incorrect_idxs = get_incorrect_prediction_idx(feed_dict, sess, model,
                                                  placeholders)
    incorrect_leg_idxs = get_incorrect_leg_idxs(feed_dict, sess, model,
                                                placeholders, leg_dict)
    predictions = sess.run(model.sentiment_model._predicted_labels(),
                           feed_dict=feed_dict)
    labels = feed_dict[placeholders[-2]]
    party_class_counts = get_party_class_counts(incorrect_leg_idxs, leg_dict,
                                                labels[incorrect_idxs])
    party_class_counts_predictions = get_party_class_counts(incorrect_leg_idxs, leg_dict,
                                                            predictions[incorrect_idxs])
    return party_class_counts, party_class_counts_predictions

def check_overlapping_days(train_days, eval_days):
    for day in eval_days:
        assert day not in train_days, "Overlap between eval and train days"
        
def check_attribute_normalization(attribute, norm_val=1.0, tolerance=1e-5):
    attribute_norm = np.linalg.norm(attribute)
    diff = np.abs(attribute_norm - norm_val)
    assert diff < tolerance, "Attribute doesn't have normalization value"

def check_unbounded_attribute(attribute, min_val=-1e20, max_val=1e20, name=" "):
    message = "Attribute {} has unbounded elements".format(name)
    num_inf = np.sum(np.isinf(attribute))
    assert num_inf==0, message
    num_nan = np.sum(np.isnan(attribute))
    assert num_nan==0, message
    num_large = np.sum(attribute > max_val)
    assert num_large==0, message
    num_small = np.sum(attribute < min_val)
    assert num_small==0, message

def find_unchanged_attribute(start_attribute, end_attribute):
    not_changed = np.unique(np.where(start_attribute==end_attribute)[0])
    return not_changed

def find_unchanged_leg_biases(start_biases, end_biases):
    unchanged_biases = find_unchanged_attribute(start_biases, end_biases)
    print("Number of unchanged legislator biases: {}".format(len(unchanged_biases)))
    return unchanged_biases

def find_unchanged_leg_embs(start_embs, end_embs):
    unchanged_embs = find_unchanged_attribute(start_embs, end_embs)
    print("Number of unchanged legislator embeddings: {}".format(len(unchanged_embs)))
    return unchanged_embs

def find_unchanged_day_embs(start_embs, end_embs):
    unchanged_embs = find_unchanged_attribute(start_embs, end_embs)
    print("Number of unchanged day embeddings: {}".format(len(unchanged_embs)))
    return unchanged_embs

def find_all_unchanged_attributes(init_attributes, end_attributes):
    unchanged_leg_embs = find_unchanged_leg_embs(init_attributes[0], end_attributes[0])
    unchanged_day_embs = find_unchanged_day_embs(init_attributes[1], end_attributes[1])
    unchanged_leg_biases = find_unchanged_leg_biases(init_attributes[2], end_attributes[2])
    return unchanged_leg_embs, unchanged_day_embs, unchanged_leg_biases

def get_unique_leg_idxs(dataset):
    inputs, labels, _ = dataset.next_batch(dataset.num_examples)
    leg_idxs = inputs[:, 0]
    return np.unique(leg_idxs)

def compare_incorrect_leg_idx_to_unchanged_biases(leg_idx_unchanged, leg_idx_incorrect):
    return (set(leg_idx_unchanged).intersection(leg_idx_incorrect))
