import numpy as np
import argparse
import pdb

def parse_model_specs(args):
    model_specs = {} # make a new dict to hold the model specifications
    model_specs["poli_dim"] = int(args["dimension"])
    model_specs["num_classes"] = int(args["num_classes"])
    model_specs["loss_ratio"] = float(args["loss_ratio"])
    model_specs["distn"] = str(args["distribution"])
    model_specs["neg_bin_param"] = float(args["neg_bin_param"])
    msg = "\nUsing a loss ratio of: {} and dimension of {}"
    msg = msg.format(model_specs["loss_ratio"], model_specs["poli_dim"])
    print(msg)
    return model_specs

def parse_model_options(args):
    opt = {}
    opt["add_leg_bias"] = True if args["add_leg_bias"]=="True" else False
    opt["use_text"] = True if args["use_text"]=="True" else False
    opt["emb_cstr"]  = None if args["emb_constraint"]=="None" else args["emb_constraint"]
    opt["bias_cstr"] = args["bias_constraint"]
    opt["use_nonlinearity"] = True if args["use_nonlinearity"]=="True" else False
    return opt

def parse_train_specs(args):
    train_dict = {}
    train_dict["learning_rate"] = float(args["learning_rate"])
    train_dict["num_iters"] = int(args["num_iterations"])
    train_dict["num_epochs"] = int(args["num_epochs"])
    train_dict["batch_size"] = int(args["batch_size"])
    train_dict["num_eval"] = int(args["num_eval"])
    train_dict["use_imputed"] =True if args["use_imputed"]=="True" else False
    return train_dict

def parse_word_embedding_filename(args):
    word_embedding_arg = args["word_embeddings"]
    if word_embedding_arg == "None":
        return None
    elif word_embedding_arg == "none":
        return None
    else:
        return word_embedding_arg
