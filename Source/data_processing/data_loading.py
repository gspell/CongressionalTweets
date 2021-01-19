import numpy as np
import pickle
import pandas as pd
import sys
import os
import pdb

from .TextData.neural_text_preprocessing import make_vocab_processor, \
                                                docs_to_integer_sequences
from .TextData.clean_text_csv import load_documents_from_csv, \
                                     remove_extra_whitespace

from .DataSet import DataSet, DataSets
#from .dataset_construction.legislator_processing import make_party_leg_dicts

# FOR USERS: PROBABLY BEST TO CHANGE THESE TO YOUR OWN
BASE_DIR  = "~/Documents/CongressionalTweets/"
DATA_DIR  = os.path.join(BASE_DIR, "TweetData/LegislatorTweetData/")
VOCAB_DIR = os.path.join(BASE_DIR, "Source/data_processing/Vocabulary/")

################################################################################
# BASIC LOADING OF PICKLE DATA
################################################################################

def load_data_from_pickle(pickle_filename):
    with open(pickle_filename, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data

def load_legislator_dict(dict_file):
    leg_dict = load_data_from_pickle(dict_file)
    return leg_dict

def load_tweet_count_matrix(count_matrix_file):
    count_matrix = load_data_from_pickle(count_matrix_file)
    return count_matrix

################################################################################
# LABEL PROCESSING/MANIPULATION
################################################################################

def conflate_labels(labels_list):
    conflated_labels = []
    for label in labels_list:
        if label < 0:
            conflated_label = -1
        elif label > 0:
            conflated_label = 1
        else:
            conflated_label = label
        conflated_labels.append(conflated_label)
    return conflated_labels

def remove_neutrals(labels_list):
    neutral_idxs = np.squeeze(np.where(labels_list==0))
    labels_no_neutrals = np.delete(labels_list, neutral_idxs)
    labels_no_neutrals[np.where(labels_no_neutrals==-1)]=0 # set neg=0, pos=1
    return neutral_idxs, labels_no_neutrals

def load_labeled_tweetData(pickled_tweet_data, num_classes=3):
    """ Note: if num_classes=3, we conflate the extreme/moderate classes.
        If num_classes=2, we conflate the classes AND remove the neutrals """
    tweets = load_data_from_pickle(pickled_tweet_data)
    leg_idxs, day_idxs, labels = tweets[:, 0], tweets[:, 1], tweets[:,2]
    neutral_idxs = None
    if num_classes==3:
        labels = conflate_labels(labels)
        labels = np.array(labels) + 1 # make the classes 0-indexed
    elif num_classes==2:
        labels = conflate_labels(labels)
        neutral_idxs, labels = remove_neutrals(labels)
        leg_idxs = np.delete(leg_idxs, neutral_idxs)
        day_idxs = np.delete(day_idxs, neutral_idxs)
    else:
        labels = np.array(labels) + 2 # make the classes 0-indexed
    return np.transpose((leg_idxs, day_idxs, labels)), neutral_idxs

def dense_to_one_hot(labels_dense, num_classes=2):
  """Convert class labels from scalars to one-hot vectors.
     Note that the class labels MUST BE INTS """
  num_labels = labels_dense.shape[0]
  print(num_labels)
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

################################################################################
# FROM COUNT MODEL DATA PROCESSING
################################################################################
def align_day_idx(trump_tweet_data, idx):
    aligned_idx = np.squeeze(np.where(trump_tweet_data[0]==idx))
    if aligned_idx>=0:
        idx_to_return = aligned_idx
    else:
        # DAMNNNN Look at that recursion
        idx_to_return = align_day_idx(trump_tweet_data, idx-1)
    return idx_to_return

def get_trump_tweet_idxs(trump_tweet_data, day_idxs):
    """ Assigns text from previous day to days without any Trump tweets """
    idx_to_get, day_idx_kept = [], []
    for idx in day_idxs:
        idx_to_get.append(align_day_idx(trump_tweet_data, idx))
        day_idx_kept.append(idx)
    return np.squeeze(idx_to_get), np.squeeze(day_idx_kept)

def get_indexed_text_seqs(trump_tweet_data, day_idxs):
    """ Returns an array of integer sequences for the Trump text and the day 
        indices for each sequence. Both are numpy arrays 
        Note: called in trump_tweet_model_runner.py and counts_runner.py """
    idx_to_get, day_idx_kept = get_trump_tweet_idxs(trump_tweet_data, day_idxs)
    text_sequences = trump_tweet_data[1][idx_to_get, :]
    return text_sequences, day_idx_kept.astype(int)

def load_day_idxs(data_dir=DATA_DIR):
    train_day_path = os.path.join(data_dir, "train_day_idxs.csv")
    val_day_path   = os.path.join(data_dir, "val_day_idxs.csv")
    test_day_path  = os.path.join(data_dir, "test_day_idxs.csv")
    train_day_idxs = np.squeeze(pd.read_csv(train_day_path, header=None).values)
    val_day_idxs   = np.squeeze(pd.read_csv(val_day_path,   header=None).values)
    test_day_idxs  = np.squeeze(pd.read_csv(test_day_path,  header=None).values)
    return (train_day_idxs, val_day_idxs, test_day_idxs)
    
def get_count_day_idxs(train_data, eval_data):
    """ I don't like this anymore! It was a mistake:
            - not all day idxs are actually in the labeled day splits
            - shouldn't have to load all data just to get day idxs anyway """
    _, train_day_idxs = make_dataset(train_data)
    _, eval_day_idxs  = make_dataset(eval_data)
    train_days = np.unique(train_day_idxs)
    eval_days  = np.unique(eval_day_idxs)
    return train_days, eval_days

def load_count_model_data():
    """ Called in trump_tweet_counts_runner.py """
    """ Note: loading all the data just to get the days seems excessive. 
        Really should have the days themselves written somewhere to be loaded """
    data_dict = load_data() # dangerous to call using only defaults...
    train_days, eval_days = get_count_day_idxs(data_dict["train_data"],
                                               data_dict["eval_data"])
    return (train_days, eval_days,
            data_dict["count_matrix"], data_dict["trump_tweet_data"])

def make_dataset(data_array):
    """ data_array has shape [num_examples, 3] with columns: 
            leg_idx, day_idx, label 
        Called in:
            trump_tweet_{model,sentiment}_runner.py and tweet_model_expanded"""
    inputs   = data_array[:, 0:2]
    labels   = data_array[:, 2]
    day_idxs = data_array[:, 1]
    idxs     = np.arange(len(labels)) # not needed, but in DataSet object
    dataset  = DataSet(inputs, labels, idxs)
    return dataset, day_idxs

################################################################################
# RIGID FILEPATH MANAGEMENT
################################################################################
def get_count_filepath(data_dir=DATA_DIR, filename="count_matrix.pickle"):
    return os.path.join(data_dir, filename)

def get_train_filepath(data_dir=DATA_DIR, filename="labeled_train.pickle"):
    return os.path.join(data_dir, filename)

def get_eval_filepath(data_dir=DATA_DIR, filename="labeled_val.pickle"):
    return os.path.join(data_dir, filename)

def get_test_filepath(data_dir=DATA_DIR, filename="labeled_test.pickle"):
    return os.path.join(data_dir, filename)

def get_trump_tweets_filepath(data_dir=DATA_DIR,
                              filename = "trumps_tweet_text.csv"):
    return os.path.join(data_dir, filename)

################################################################################
#
################################################################################
def load_imputed_data(data_dir=DATA_DIR, filename="imputed_train.pickle",
                      num_classes=3):
    imputed_train_filepath = os.path.join(data_dir, filename)
    imputed_data, _ = load_labeled_tweetData(imputed_train_filepath, num_classes)
    return imputed_data

def load_data(data_dir=DATA_DIR, vocab_save_dir=VOCAB_DIR, num_classes=3,
              max_length=200, min_freq=2, do_eval=True, use_imputed=False):
    data = {}
    train_filepath        = get_train_filepath(data_dir)
    eval_filepath         = get_eval_filepath(data_dir)
    test_filepath         = get_test_filepath(data_dir)
    counts_filepath       = get_count_filepath(data_dir)
    trump_tweets_filepath = get_trump_tweets_filepath(data_dir)

    data_day_idxs    = load_day_idxs(data_dir)
    data["train"], _ = load_labeled_tweetData(train_filepath, num_classes)
    data["eval"],  _ = load_labeled_tweetData(eval_filepath,  num_classes)
    data["test"],  _ = load_labeled_tweetData(test_filepath,  num_classes)
    data["counts"]   = load_tweet_count_matrix(counts_filepath)
    data["trump_tweet_data"] = load_trump_tweets_data(trump_tweets_filepath,
                                                      max_length=max_length,
                                                      min_freq=min_freq,
                                                      save_dir=vocab_save_dir)
    if use_imputed is True:
        imputed       = load_imputed_data(data_dir, num_classes=num_classes)
        imputed       = get_imputed_for_missing_legis(data["train"], imputed)
        data["train"] = np.vstack((data["train"], imputed))
        
    data["train_days"], data["eval_days"], data["test_days"] = data_day_idxs[:]
    return data

def get_imputed_for_missing_legis(train_data, imputed_data):
    legs_in_train = np.unique(train_data[:, 0])
    imputed_missing_legislators = []
    for example in imputed_data:
        if example[0] not in legs_in_train:
            imputed_missing_legislators.append(example)
    pdb.set_trace()
    return np.stack(imputed_missing_legislators)

"""
def party_split_dataset(full_leg_dict, tweet_data_array):
    rep_leg_dict, dem_leg_dict = make_party_leg_dicts(full_leg_dict)
    rep_leg_ids = list(rep_leg_dict.keys())
    dem_leg_ids = list(dem_leg_dict.keys())
    tweet_data_shape = np.shape(tweet_data_array)
    assert tweet_data_shape[1]==3 #ensure the axes of the array are correct
    # first column is the legislator ids
    dem_idxs = np.isin(tweet_data_array[:,0], dem_leg_ids)
    rep_idxs = np.isin(tweet_data_array[:,0], rep_leg_ids)
    dem_tweet_array = tweet_data_array[dem_idxs]
    rep_tweet_array = tweet_data_array[rep_idxs]
    return (rep_leg_dict, rep_tweet_array), (dem_leg_dict, dem_tweet_array)
"""

################################################################################
# TEXT UTILS
################################################################################
def make_text_int_seqs(text, max_length, min_freq, save_dir=VOCAB_DIR):
    """ This should probably just go to TextData/neural_text_processing.py """
    _, vocab = make_vocab_processor(text, max_length, min_freq, save_dir)
    text_int_sequences = docs_to_integer_sequences(text, vocab)
    return text_int_sequences

def load_trump_tweets_data(filename, max_length=500, min_freq=2,
                           save_dir=VOCAB_DIR):
    trump_tweet_dataframe = pd.read_csv(filename)
    trump_tweet_seqs = make_text_int_seqs(trump_tweet_dataframe["text"],
                                          max_length, min_freq, save_dir)
    date_idx = trump_tweet_dataframe.date_idx.values
    return (date_idx, trump_tweet_seqs)

