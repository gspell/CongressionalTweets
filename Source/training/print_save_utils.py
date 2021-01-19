import numpy as np
import tensorflow as tf
import pickle
import os
import decimal
import sys
import shutil
sys.path.append("../")

def print_mae_metrics(i, train_loss, train_mae, eval_loss, eval_mae):
    msg = ("Step {}, Train Loss: {:.2f}, Train MAE: {:.3f}, " +
          "Eval MAE: {:.3f} Eval Loss: {:.2f}")
    print(msg.format(i, train_loss, train_mae, eval_mae, eval_loss))
    return None

def print_loss_metrics(i, train_count_loss, train_sent_loss,
                       eval_count_loss, eval_sent_loss):
    x = (i, train_count_loss, train_sent_loss, eval_count_loss, eval_sent_loss)
    msg = "Step {}, Tr-C: {:.1f}, Tr-S: {:.1f}, Eval-C: {:.1f}, Eval-S: {:.1f}"
    print(msg.format(*x))
    return None

def print_final_metrics(train_metrics, eval_metrics):
    print("\tFinal Metrics: \t MAE \t Acc \t  Loss")
    print("\t"+"-"*40)
    print("\tTraining: \t {:.3f} \t {:.3f} \t  {:.2f}".format(*train_metrics))
    print("\tEval:     \t {:.3f} \t {:.3f} \t  {:.2f}".format(*eval_metrics))
    return None

def pickle_data(data, filename):
    with open(filename, "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    return None

def make_attribute_filenames(save_dir):
    filenames = {}
    filenames["leg_embs"] = os.path.join(save_dir, "leg_embs.pickle")
    filenames["day_embs"] = os.path.join(save_dir, "day_embs.pickle")
    filenames["leg_bias"] = os.path.join(save_dir, "leg_bias.pickle")
    return filenames

def pickle_legislator_attributes(leg_embs, day_embs, leg_biases, save_dir):
    filenames = make_attribute_filenames(save_dir)
    pickle_data(leg_embs, filenames["leg_embs"])
    pickle_data(day_embs, filenames["day_embs"])
    pickle_data(leg_biases, filenames["leg_bias"])

def pickle_text_attributes(text_attributes, save_dir):
    """ text_attributes should now be a dict! """
    text_params_filename = os.path.join(save_dir, "text_model_params.pickle")
    pickle_data(text_attributes, text_params_filename)

def save_mae_to_csv(train_mae, eval_mae, save_dir, filename):
    results_filename = os.path.join(save_dir, filename)
    f = open(results_filename, "ab")
    MAEs = list([tr_mae, ev_mae])
    np.savetxt(f, np.expand_dims(MAEs, axis=0), fmt="%.3f", delimiter=", ")
    f.close()

def save_count_loss(tr_count_loss, ev_count_loss, save_dir, filename):
    results_filename = os.path.join(save_dir, filename)
    f = open(results_filename, "ab")
    count_losses = list([tr_count_loss, ev_count_loss])
    np.savetxt(f, np.expand_dims(count_losses, axis=0),
               fmt="%.2f", delimiter=", ")
    f.close()

def save_eval_metrics(eval_metrics, save_dir, filename="eval_metrics.csv"):
    """ Eval metrics are tuple: (eval_mae, eval_acc, eval_loss) """
    results_filename = os.path.join(save_dir, filename)
    f = open(results_filename, "ab")
    np.savetxt(f, np.expand_dims(eval_metrics, axis=0),
               fmt="%.2f", delimiter=", ")
    f.close()
    
def save_metrics_to_csv(tr_loss, ev_loss, tr_mae, ev_mae, tr_count_loss,
                        ev_count_loss, save_dir, filename):
    results_filename = os.path.join(save_dir, filename)
    f = open(results_filename, "ab") # this will append to the file
    metrics = list([tr_loss, ev_loss, tr_mae, ev_mae,
                    tr_count_loss, ev_count_loss])
    np.savetxt(f, np.expand_dims(metrics, axis=0), fmt="%.2f", delimiter=", ")
    f.close()

def make_dimension_save_dir(root_dir, dim):
    dim_save_dir = os.path.join(root_dir, str(dim))
    if os.path.isdir(dim_save_dir):
        print("Save Directory already exists. Will append to it")
        #print("Save directory already exists. Deleting old one")
        #shutil.rmtree(dim_save_dir)
    else:
        print("Making save directory {}".format(dim_save_dir))
        os.makedirs(dim_save_dir)
    
    return dim_save_dir

def make_dim_ratio_save_dir(root_dir, dim, ratio):
    dim_ratio_string = str(dim) + "_" + float_to_str(ratio)
    dim_ratio_dir = os.path.join(root_dir, dim_ratio_string)
    if os.path.isdir(dim_ratio_dir):
        print("Save directory already exists. Deleting old one")
        shutil.rmtree(dim_ratio_dir)
    print("Making save directory {}".format(dim_ratio_dir))
    os.makedirs(dim_ratio_dir)
    return dim_ratio_dir

def float_to_str(f):
    ctx = decimal.Context()
    ctx.prec = 20
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')
