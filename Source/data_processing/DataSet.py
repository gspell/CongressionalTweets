from __future__ import print_function
import numpy as np
import random
from random import shuffle

class DataSet(object):

  def __init__(self, inputs, labels, idxs):
    self._num_examples = np.shape(inputs)[0]
    self._inputs = np.array(inputs)
    self._labels = labels
    self._idxs = idxs
    self._epochs_completed = 0
    self._index_in_epoch = 0
  
  @property
  def inputs(self):
    return self._inputs
  
  @property
  def idxs(self):
    return self._idxs
    
  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed
  
  def _finish_epoch(self):
    self._epochs_completed += 1
  
  def _shuffle_dataset(self, permutation):
    combined_data_items = list(zip(self._inputs, self._labels, self._idxs))
    shuffled_data = shuffle_data_indices(combined_data_items, permutation)
    x, labels, idxs = (np.squeeze(data_item) for data_item in zip(*shuffled_data)) #unzip list
    return x, labels, idxs

  def _start_next_epoch(self, batch_size):
    permutation = get_shuffling_permutation(self._num_examples)
    self._inputs, self._labels, self._idxs = self._shuffle_dataset(permutation)
    self._index_in_epoch = 0

  def _get_end_idx(self, batch_size):
    self._index_in_epoch += batch_size
    if self._index_in_epoch >= self._num_examples:
      self._finish_epoch()
      return self._num_examples
    else:
      return self._index_in_epoch

  def _get_batch_idxs(self, batch_size):
    start_idx = self._index_in_epoch
    end_idx = self._get_end_idx(batch_size)
    if start_idx == end_idx:
      print("OOPS, start_idx = {} and end_idx = {}".format(start_idx, end_idx))
    return start_idx, end_idx
  
  def next_batch(self, batch_size):
    if self._index_in_epoch >= self._num_examples:
      self._start_next_epoch(batch_size)
    start, end = self._get_batch_idxs(batch_size)
    return self._inputs[start:end], self._labels[start:end], self._idxs[start:end]

class DataSets(object):
  pass

def shuffle_data_indices(data, permutation=None):
  if permutation is None:
    # np.random.shuffle(data) # Does this in-place. Not necessarily desirable!
    permutation = get_shuffling_permutation(len(data))
  data = [data[i] for i in permutation] # This is for if data is of type list
    # If data is an np.array, can do data[perm] directly
    # What are the advantages of one over the other above????
  return data

def get_shuffling_permutation(permutation_length):
  sequence = np.arange(permutation_length)
  np.random.shuffle(sequence)
  return sequence
