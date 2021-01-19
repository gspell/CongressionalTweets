#!/bin/bash
trap 'exit 130' INT

# REPLACE WITH YOUR OWN DIRECTORY STRUCTURE IF NEEDED
BASE_DIR="~/Documents/CongressionalTweets/"
RESULTS_DIR=$BASE_DIR"Results/"

LR=0.0001
SEED=0

# FOR COUNT ONLY PREDICTIVE RESULTS

echo "\nRunning Neg-Bin model, only counts, with text"
# 1) Get Negative Binomial, Only counts, text results
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=75 \
       --batch_size=128 \
       --distribution=negative_binomial \
       --loss_ratio=1 \
       --save_dir=$RESULTS_DIR"NegBin/Text/" \
       --neg_bin_param=0.1

echo "\nRunning Neg-Bin model, only counts, no text"
# 2) Get Negative Binomial, Only counts,no text results
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=75 \
       --batch_size=128 \
       --distribution=negative_binomial \
       --loss_ratio=1 \
       --neg_bin_param=0.1 \
       --save_dir=$RESULTS_DIR"NegBin/NoText/" \
       --use_text=False

echo "\nRunning poisson model, only counts, with text"
# 3) Get Poisson, Only counts, text results
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=100 \
       --batch_size=128 \
       --distribution=poisson \
       --save_dir=$RESULTS_DIR"Poisson/Text/" \
       --loss_ratio=1

echo "\nRunning poisson model, only counts, no text"
# 4) Get Poisson, Only counts,no text results - note how many iterations needed!!
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=2000 \
       --batch_size=128 \
       --distribution=poisson \
       --loss_ratio=1 \
       --save_dir=$RESULTS_DIR"Poisson/NoText/" \
       --use_text=False

# FOR SENTIMENT ONLY PREDICTIONS

echo "\nRunning sentiment only model, with text and bias"
# 1) Sentiment - with text, include leg bias
# Can achieve same MAE/Acc with 2000 iterations, but lower loss
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=150 \
       --batch_size=128\
       --distribution=negative_binomial \
       --loss_ratio=0 \
       --save_dir=$RESULTS_DIR"NegBin/Text/" \
       --neg_bin_param=0.1

echo "\nRunning sentiment only model, without text and with bias"
# 2) Sentiment - without text, include leg bias - again, SO MANY iterations
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=1000 \
       --batch_size=128 \
       --distribution=negative_binomial \
       --loss_ratio=0 \
       --save_dir=$RESULTS_DIR"NegBin/NoText/" \
       --neg_bin_param=0.1 \
       --use_text=False

echo "\nRunning sentiment only model, with text and without bias"
# 3) Sentiment - with Text, but no leg bias
# This should be run with the same number of iterations as Text, Bias, which is most fair. They are very, VERY comparable in performance
 python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=150 \
       --batch_size=128 \
       --distribution=negative_binomial \
       --loss_ratio=0 \
       --save_dir=$RESULTS_DIR"NegBin/Text/NoBias/" \
       --neg_bin_param=0.1 \
       --add_leg_bias=False

 echo "\nRunning sentiment only model, without text and without bias"
 # 4) Sentiment - no Text, and no leg bias
 # Essentially can't improve on eval performance, no matter how many iterations
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=3000 \
       --batch_size=128 \
       --distribution=negative_binomial \
       --loss_ratio=0 \
       --save_dir=$RESULTS_DIR"NegBin/NoText/NoBias/" \
       --neg_bin_param=0.1 \
       --use_text=False \
       --add_leg_bias=False


# THE JOINT MODEL. CHOOSING LOSS RATIO=0.03.
echo "\nRunning joint model, with text. LR=0.03"
# 1) Neg-Bin, with leg bias, text
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=200 \
       --batch_size=128 \
       --distribution=negative_binomial \
       --loss_ratio=0.03 \
       --save_dir=$RESULTS_DIR"NegBin/Text/" \
       --neg_bin_param=0.1 \
       --use_text=True

echo "\nRunning joint model, without text. LR=0.03"
# 2) Neg-Bin, with leg bias, no text
python trump_tweet_model_runner.py \
       --dimension=2 \
       --num_epochs=1500 \
       --batch_size=128 \
       --distribution=negative_binomial \
       --loss_ratio=0.03 \
       --neg_bin_param=0.1 \
       --save_dir=$RESULTS_DIR"NegBin/NoText/" \
       --use_text=False
