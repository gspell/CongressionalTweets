import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os
import argparse
import pdb
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
sys.path.append("../")

import model_utils as utils
import model_analysis as analysis
import print_save_utils as print_save
from data_processing.data_loading import load_data, load_legislator_dict, \
                                         make_dataset, get_indexed_text_seqs, \
                                         party_split_dataset, load_labeled_tweetData
from model.trump_tweet_model_OOP import TrumpTweetModel
from trump_tweet_model_runner_base import TweetModelRunner
import model_config_utils as config_utils
import time                        

class TrumpTweetModelRunner(TweetModelRunner):
    def __init__(self, model_specs, model_options, nums_tweets, train_options):
        self.nums_tweets = nums_tweets
        super().__init__(model_specs, model_options, train_options)
        self.mae, self.acc = self.MAE(), self.ACC()

    def run_word_emb_init(self, word_emb_filename):
        init_msg = "Initializing word embeddings from {}"
        print(init_msg.format(word_emb_filename))
        embs = utils.load_word_embeddings(word_emb_filename)
        init_op = self.model.text_model.SWEM_model.word_embs.assign(embs)
        self.sess.run(init_op)
        return None
    
    def init_word_embeddings(self, word_emb_filename=None):
        if word_emb_filename is not None:
            self.run_word_emb_init(word_emb_filename)
        else:
            print("Using randomly initialized word embeddings")

    def get_sentiment_placeholders(self):
        sentiment_PHs = utils.get_sentiment_placeholders()
        self.lbld_legs, self.lbld_days, self.labels = sentiment_PHs
        return sentiment_PHs

    def get_count_placeholders(self, model_specs):
        num_legis, num_days = model_specs["num_legis"], model_specs["num_days"]
        count_PHs = utils.get_count_placeholders(num_legis, num_days)
        self.true_counts, self.count_days, self.count_legs = count_PHs
        return count_PHs
    
    def get_placeholders(self, seq_length = 200):
        sentiment_PHs = self.get_sentiment_placeholders()
        count_PHs = self.get_count_placeholders(self.model_specs)
        self.in_text = utils.get_text_placeholders(seq_length, name="in_text")
        self.count_text = utils.get_text_placeholders(seq_length, name="count_text")
        placeholders = sentiment_PHs + count_PHs + (self.in_text, self.count_text)
        return placeholders

    def get_sentiment_feed_dict(self, x, y, trump_tweets):
        legs, days = x[:, 0], x[:,1] # the legislator and day indices, unpacked
        feed_dict = {self.lbld_legs:legs, self.lbld_days:days, self.labels:y}
        sent_text_seqs = self.get_text_seqs(trump_tweets, days)
        feed_dict.update({self.in_text:sent_text_seqs})
        return feed_dict

    def get_count_feed_dict(self, counts, day_idxs, trump_tweets, leg_idxs):
        feed_dict = {self.count_legs:leg_idxs, self.count_days:day_idxs}
        feed_dict.update({self.true_counts:counts})
        count_text_seqs = self.get_text_seqs(trump_tweets, day_idxs)
        feed_dict.update({self.count_text:count_text_seqs})
        return feed_dict

    def get_text_seqs(self, trump_tweets, day_idxs):
        text_seqs, text_idxs = get_indexed_text_seqs(trump_tweets, day_idxs)        
        assert np.all(text_idxs==day_idxs)
        return text_seqs
    
    def get_feedDict(self, counts, x, y, trump_tweets, legs_to_use, count_days):
        days, legs = x[:, 1], legs_to_use
        sentiment_FD = self.get_sentiment_feed_dict(x, y, trump_tweets)
        count_FD = self.get_count_feed_dict(counts, count_days, trump_tweets, legs)
        feed_dict = {**sentiment_FD, **count_FD} # merge dicts, unraveling
        return feed_dict
    
    def build_model(self):
        return TrumpTweetModel(self.model_specs, self.model_options,
                               self.lbld_legs, self.lbld_days, self.labels,
                               self.in_text, self.count_legs, self.count_text,
                               self.count_days)
        
    def loss(self):
        print("\nMaking loss-op")
        return self.model.get_total_loss(self.true_counts, self.nums_tweets)

    def MAE(self):
        print("\nMaking MAE op")
        return self.model.get_error_metric()
    
    def ACC(self):
        print("\nMaking accuracy op")
        return self.model.get_accuracy()

    def prepare_training(self, train_data, eval_data, counts, trump_tweets,
                         train_days, eval_days, train_legs):
        x, y = train_data.inputs, train_data.labels
        self.full_train_feed_dict = self.get_feedDict(
            counts, x, y, trump_tweets, train_legs, train_days)
        x, y = eval_data.inputs, eval_data.labels
        self.eval_feed_dict = self.get_feedDict(
            counts, x, y, trump_tweets, np.arange(468), eval_days)
        self.train_count_feed_dict = self.get_count_feed_dict(
            counts, train_days, trump_tweets, train_legs)
        self.eval_count_feed_dict = self.get_count_feed_dict(
            counts, eval_days, trump_tweets, np.arange(468))
        
    def run_train_step(self, train_data, trump_tweets, counts,
                       train_days, batch_size=200, train_legs=None):
        x, y, _ = train_data.next_batch(batch_size=batch_size) #x-inputs, y-labels
        feed_dict = self.get_feedDict(counts, x, y, trump_tweets,
                                      train_legs, train_days)
        _, step_loss = self.sess.run([self.train_op, self.loss], feed_dict)
        other_losses = [self.model.count_loss, self.model.sentiment_loss]
        count_loss, sent_loss = self.sess.run(other_losses, feed_dict)
        
        self.perform_attribute_updates() # impose constraints, if needed

    def compute_iters_per_epoch(self, train_data, batch_size):
        return np.ceil(train_data.num_examples / batch_size).astype(int)

    def run_training_loop(self, train_data, counts, trump_tweets, train_days,
                          eval_days, train_legs, results_dir):
        batch_size = self.train_options["batch_size"]
        num_epochs = self.train_options["num_epochs"]
        print("\nCommencing training loop\n")
        iters_per_epoch = self.compute_iters_per_epoch(train_data, batch_size)
        num_iters_total = iters_per_epoch * num_epochs
        print("Need {} iterations per epoch\n".format(iters_per_epoch))
        msg = "Will perform {} iterations total, with batch size of {}\n"
        print(msg.format(num_iters_total, batch_size))
        time_start = time.perf_counter()
        for i in range(num_iters_total+1):
            self.run_train_step(train_data, trump_tweets, counts, train_days,
                                batch_size, train_legs)
            if i % self.train_options["num_eval"] == 0:
                self.at_each_eval(i, results_dir)
        time_end = time.perf_counter()
        print("Took {} seconds to train".format((time_end - time_start)))
        
    def at_each_eval(self, step, results_dir):
        tr_loss, tr_mae = self.run_loss_and_mae(self.full_train_feed_dict)
        eval_loss, eval_mae = self.run_eval(self.eval_feed_dict)
        train_count_loss = self.run_count_loss(self.train_count_feed_dict)
        eval_count_loss = self.run_count_loss(self.eval_count_feed_dict)
        print_save.print_mae_metrics(step, tr_loss, tr_mae, eval_loss, eval_mae)
        print_save.save_metrics_to_csv(tr_loss, eval_loss, tr_mae, eval_mae,
                                       train_count_loss, eval_count_loss,
                                       results_dir, "training_metrics.csv")
        
    def run_count_loss(self, feed_dict):
        count_loss = self.sess.run(self.model.count_loss, feed_dict)
        return count_loss
    
    def run_loss_and_mae(self, feed_dict):
        loss, mae = self.sess.run([self.loss, self.mae], feed_dict)
        return loss, mae
    
    def run_eval(self, feed_dict):
        eval_loss, eval_mae = self.sess.run([self.loss, self.mae], feed_dict)
        return eval_loss, eval_mae
    
    def get_final_metrics(self, feed_dict):
        metric_list = [self.mae, self.acc, self.loss]
        mae, acc, loss = self.sess.run(metric_list, feed_dict)
        return (mae, acc, loss)
        
def run_model(model_specs, model_options, word_embs, data_dict, train_legs,
              results_dir, train_options, seed=0, do_eval=True):
    start_time = time.time()
    utils.set_random_seeds(seed=seed)
    num_legis, num_days, nums_tweets = utils.get_countQuantities(data_dict["counts"])
    text_params = {"word_emb_size":300, "text_length":200, "vocab_size":2783}
    model_specs.update({"num_legis":num_legis, "num_days":num_days})
    model_specs.update({"text_params":text_params})
    model_runner = TrumpTweetModelRunner(model_specs, model_options,
                                         nums_tweets, train_options)

    train_dataset, _ = make_dataset(data_dict["train"])
    train_days       = data_dict["train_days"]
    if do_eval is False:
        print("\nLoading the test (not validation)")
        eval_dataset, _ = make_dataset(data_dict["test"])
        eval_days       = data_dict["test_days"]
    else:
        eval_dataset, _ = make_dataset(data_dict["eval"])
        eval_days       = data_dict["eval_days"]
    analysis.check_overlapping_days(train_days, eval_days)

    model_runner.init_global_vars()
    init_attributes = model_runner.get_leg_model_attributes()
    model_runner.init_word_embeddings(word_embs)
    #train_unique_legs = get_unique_leg_idxs(train_dataset)
    #sess = init_leg_biases(sess, model, train_unique_legs, num_legis=468)
    
    model_runner.prepare_training(train_dataset, eval_dataset,
                                  data_dict["counts"],
                                  data_dict["trump_tweet_data"], train_days,
                                  eval_days, train_legs)
    
    model_runner.run_training_loop(train_dataset, data_dict["counts"],
                                   data_dict["trump_tweet_data"], train_days,
                                   eval_days, train_legs, results_dir)
    
    print("\nHave left the training loop")
    end_time = time.time()
    print("\nTotal time to run model {:.3f}".format(end_time-start_time))
    
    ############################################################################
    # Make sentiment predictions for all tweets
    ############################################################################
        
    final_train_feed_dict = model_runner.full_train_feed_dict
    final_eval_feed_dict = model_runner.eval_feed_dict
    tr_metrics, ev_metrics = model_runner.finish_training(final_train_feed_dict,
                                                          final_eval_feed_dict)
    print_save.print_final_metrics(tr_metrics, ev_metrics)
    final_attributes = model_runner.get_leg_model_attributes()
    text_attributes = model_runner.get_text_model_attributes() # now dict
    analysis.find_all_unchanged_attributes(init_attributes, final_attributes)
    model_runner.check_satisfied_constraints()
    model_runner.sess.close()
    return final_attributes, text_attributes, ev_metrics

def maybe_party_split(party, leg_dict, train_data):
    train_legis = list(leg_dict.keys())
    reps_data, dems_data = party_split_dataset(leg_dict, train_data)
    if party=="R":
        train_legis = list(reps_data[0].keys())
        train_data = reps_data[1]
    elif party=="D":
        train_legis = list(dems_data[0].keys())
        train_data = dems_data[1]
    return train_legis, train_data

def main():
    parser = argparse.ArgumentParser(description="Tweet Model Arguments")
    parser.add_argument('--dimension', default=2) # political_dim
    parser.add_argument('--seed', default=0)
    parser.add_argument('--num_classes', default=3) # number of ordinal classes
    parser.add_argument('--save_dir', default="../../Results/default")
    parser.add_argument('--data_dir', default="../../TweetData/LegislatorTweetData/")
    parser.add_argument('--model', default="ordinal")
    parser.add_argument('--loss_ratio', default=0.03)
    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--num_epochs', default=500)
    parser.add_argument('--num_iterations', default=5000)
    parser.add_argument('--batch_size', default=200)
    parser.add_argument('--num_eval', default=100)
    parser.add_argument('--use_text', default="True")
    parser.add_argument('--add_leg_bias', default="True")
    parser.add_argument('--word_embeddings', default=None)
    parser.add_argument('--party', default="both")
    parser.add_argument('--use_nonlinearity', default=False)
    parser.add_argument('--emb_constraint', default=None)
    parser.add_argument('--bias_constraint', default=None)
    parser.add_argument('--distribution', default="poisson")
    parser.add_argument('--neg_bin_param', default=1)
    parser.add_argument('--use_imputed', default=False)
    args = vars(parser.parse_args())

    seed = int(args["seed"])
    word_emb_path = config_utils.parse_word_embedding_filename(args)
    model_specs = config_utils.parse_model_specs(args) # now a dict
    model_options = config_utils.parse_model_options(args) # now a dict
    train_options= config_utils.parse_train_specs(args)
    dim_ratio_save_dir = print_save.make_dim_ratio_save_dir(args["save_dir"],
                                                        model_specs["poli_dim"],
                                                        model_specs["loss_ratio"])
    data_dict = load_data(data_dir=args["data_dir"],
                          vocab_save_dir=dim_ratio_save_dir,
                          num_classes=model_specs["num_classes"],
                          max_length=200, min_freq=2, do_eval=True,
                          use_imputed=train_options["use_imputed"])
    leg_dict_path = os.path.join(args["data_dir"], "legislator_dict.pickle")
    leg_dict      = load_legislator_dict(leg_dict_path)
    train_legis, data_dict["train"] = maybe_party_split(args["party"],
                                                            leg_dict,
                                                            data_dict["train"])
    
    
    model_attributes, text_attributes, eval_metrics = run_model(
        model_specs, model_options, word_emb_path, data_dict, train_legis,
        dim_ratio_save_dir, train_options, seed, do_eval=False)
    print_save.save_eval_metrics(eval_metrics, dim_ratio_save_dir,
                                 "eval_metrics.csv")
    print_save.pickle_legislator_attributes(*model_attributes,
                                            dim_ratio_save_dir)
    print_save.pickle_text_attributes(text_attributes, dim_ratio_save_dir)


if __name__== "__main__":
    main()
