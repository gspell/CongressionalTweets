import numpy as np
import pickle
import nltk
import csv
import os
import string
import re
from string import punctuation, digits
import os

from tensorflow.contrib import learn

from sklearn.feature_extraction.text import TfidfVectorizer

import collections
from collections import defaultdict
from collections import Counter

def write_dict_to_csv(dict_to_write, filename):
  with open(filename, "w") as f:
    writer = csv.writer(f)
    writer.writerows(dict_to_write.items())

def pickle_dict(dict_to_pickle, filename):
  with open(filename, "wb") as pickle_file:
    pickle.dump(dict_to_pickle, pickle_file)
    
def get_vocab_dict_from_processor(vocab_processor, save_dir):
  vocab_dict = vocab_processor.vocabulary_._mapping
  vocab_dict_filename = os.path.join(save_dir, "vocab_dict.csv")
  vocab_pickle_filepath = os.path.join(save_dir, "vocab.pickle")
  write_dict_to_csv(vocab_dict, vocab_dict_filename)
  pickle_dict(vocab_dict, vocab_pickle_filepath)
  return vocab_dict

def restore_vocab_processor(max_length, min_freq,
                            save_dir, filename="vocab_processor.data"):
  VocabProcessor  = learn.preprocessing.VocabularyProcessor
  vocab_processor = VocabProcessor(max_document_length=max_length,
                                   min_frequency = min_freq)
  processor_path  = os.path.join(save_dir, filename)
  vocab_processor = VocabProcessor.restore(processor_path)
  return vocab_processor
  
def make_vocab_processor(text, max_length, min_freq, save_dir):
  VocabProcessor = learn.preprocessing.VocabularyProcessor
  vocab_processor = VocabProcessor(max_document_length = max_length,
                                   min_frequency = min_freq)
  vocab_processor.fit(text)
  vocab_dict = get_vocab_dict_from_processor(vocab_processor, save_dir)
  processor_filename = os.path.join(save_dir, "vocab_processor.data")
  vocab_processor.save(processor_filename)
  return vocab_dict, vocab_processor

def docs_to_integer_sequences(documents, vocab_processor):
  return np.array(list(vocab_processor.transform(documents)))

""" These are kind of from wayyyy back in the day (not used anymore) """

def build_vocab(doc_list):
    vocab = defaultdict(float)
    for doc in doc_list:
        words = doc.split()
        for word in words:
            vocab[word] += 1
    wordtoix = defaultdict(float)
    ixtoword = defaultdict(float)
    
    count = 1 #start at 1 so the 0 can be for the padded token
    for w in vocab.keys():
        wordtoix[w] = count
        ixtoword[count] = w
        count +=1
    return wordtoix, ixtoword

def get_idx_from_sent(sent, wordtoix, max_l):
    x = []
    words = sent.split()
    for word in words:
        x.append(wordtoix[word])
    while len(x) < max_l:
        x.append(0)
    return x

def make_idx_data(bills, wordtoix, max_l):
    x = []
    for bill in bills:
        sent = get_idx_from_sent(bill, wordtoix, max_l)
        x.append(sent)
    return x
    
def preprocess(text):
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for bill in text:
        sentence = sent_detector.tokenize(bill)
        result = ''
        for s in sentence:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    max_l = len(max(X, key=len))
    return X, max_l
