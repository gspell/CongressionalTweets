import pickle
import csv
import os
import string
import re

def load_documents_from_csv(filename):
  print("\nLoading documents from {}".format(filename))
  with open(filename) as f:
    reader = csv.reader(f)
    docs = f.readlines()
  del(f)
  return docs

def remove_extra_whitespace(bills):
    """ Removes trailing new line characters and extra spaces around punctuation """
    X = [bill.split()[:] for bill in bills]
    X = [' '.join(bill) for bill in X]
    return X

def process_apostrophes(string):
  # split contractions ending in 's. Note Glove has 's token
  string = re.sub("\'s", " \'s", string)
  # remove double quotation marks
  string = re.sub("\'{2,}", " ", string)
  # pad whitespace around trailing apostrophe
  string = re.sub("\'\s", " \' ", string)
  return string

def pad_whitespace_around_punctuation(string):
  string = re.sub("\.", " . ", string) # pad whitespace around periods
  string = re.sub(",", " , ", string)
  string = re.sub(";", " ; ", string)
  string = re.sub(":", " : ", string)
  string = re.sub("\?", " ? ", string)
  string = re.sub("\!", " ! ", string)
  return string
   
def clean_str(string):
  #string = re.sub("\!\?", ".", string) # turn !? into periods
  # retain alpha-numerics , . ; : ' ? !
  string = re.sub("[^A-Za-z0-9,;:\'\.\?\!]", " ", string) 
  string = process_apostrophes(string)
  string = pad_whitespace_around_punctuation(string)
  string = re.sub("\s{2,}", " ", string) # multiple spaces to single space
  return string.strip().lower()

def save_string_list_as_csv(list_of_strings, output_filename):
  with open(output_filename, "w") as f:
    writer = csv.writer(f, delimiter="\n")
    for string in list_of_strings:
      writer.writerow([string])

def main():
  filename_to_process = input("What is the file path to load and process? ")
  documents = load_documents_from_csv(filename_to_process)
  documents = remove_extra_whitespace(documents)
  documents = [clean_str(document) for document in documents]
  filename_to_save = input("What is the file path to save cleaned documents? ")
  save_string_list_as_csv(documents, filename_to_save)

if __name__=="__main__":
  main()
