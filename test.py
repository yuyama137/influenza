from random import shuffle
import sys
import time
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import numpy as np
import model
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from functools import partial
import math
import scipy
import json

import util
import model
plt.rcParams["font.size"] = 18

SEED = 42
util.fix_seed(SEED)

def make_dataloader(data, input_step, predict_step, use_clm, use_state, batch_size=1):
    dataset = util.ILIDataset(data, input_step, predict_step, use_clm, use_state)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader

def main():
    parser = argparse.ArgumentParser(description='Test Voice Transformer Network')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_path', '-model', type=str, help='root of model')
    parser.add_argument('--outpath', '-out', default="./checkout/test", type=str, help='root of output directory')
    parser.add_argument('--check_state', '-state', default="Alabama", type=str)
    parser.add_argument('--epoch_num', '-epn', default=9, type=int, help="学習済みモデルのエポック数")
    parser.add_argument('--dbg', '-dbg', action="store_true")
    args = parser.parse_args()

    model_path = args.model_path
    outpath = args.outpath
    check_state = args.check_state
    epoch_num = args.epoch_num

    config_path = os.path.join(model_path, "config/config.json")

    with open(config_path) as f:
        config = json.load(f)

    _, model_name = os.path.split(model_path)
    train_data_root = config["train_data_root"]
    test_data_root = config["test_data_root"]
    input_step = config["input_step"]
    predict_step = config["predict_step"]
    use_clm = config["use_clm"]
    d_model = config["d_model"]
    attn_type = config["attn_type"]
    N_enc = config["N_enc"]
    N_dec = config["N_dec"]
    h_enc = config["h_enc"]
    h_dec = config["h_dec"]
    ff_hidnum = config["ff_hidnum"]
    hid_pre = config["hid_pre"]
    hid_post = config["hid_post"]
    dropout_pre = config["dropout_pre"]
    dropout_post = config["dropout_post"]
    dropout_model = config["dropout_model"]
    use_state = config["use_state"]
    in_dim = len(use_clm)
    
    par_path = os.path.join(model_path, "model/{}_epoch.model".format(epoch_num))

    #######################################
    ############ Assert ###################
    #######################################

    if use_state != "all":
        if use_state[0] != check_state:
            assert False, "please set checkstate same as trained state"
    
    #########################################################################
    device = torch.device("cuda:{}".format(args.gpu)) if args.gpu >= 0 else torch.device("cpu")

    if use_state == "all":
        model_name = model_name+"_{}".format(check_state)
    util.set_directories(outpath, model_name, ["scatter", "forcast", "log"], args.dbg)
    log_path = os.path.join(outpath, model_name, "log")
    scatter_path = os.path.join(outpath, model_name, "scatter")
    forcast_path = os.path.join(outpath, model_name, "forcast")

    logger = util.Logger(log_path, "log", args_dict=config)

    _, test_data = util.load_data(train_data_root, test_data_root)

    test_set, test_loader = make_dataloader(test_data, input_step, predict_step, use_clm, [check_state], batch_size=1)
    all_data = test_set.data[check_state][:,0]

    net = model.Transformer(device, d_model, in_dim, attn_type, N_enc, N_dec, h_enc, h_dec, ff_hidnum, hid_pre, hid_post, dropout_pre, dropout_post, dropout_model)
    
    # load parameter
    if device == torch.device("cpu"):
        net.load_state_dict(torch.load(par_path, map_location=torch.device("cpu")))
    else:
        net.load_state_dict(torch.load(par_path))
    # net = net.to(device)

    net.eval()

    # predict
    # 出力するもの
    #  - scatter plot
    #  - 1時点先を予測し続けたplot
    pred_list = np.zeros((1, predict_step))# (1, pred_len(1時点先, 2時点先))
    for iter, (x, y, tgt, key) in enumerate(test_loader):
        x, y, tgt = x.to(device), y.to(device), tgt.to(device)

        tgt = tgt[:,:,0]

        out = net.generate(x, tgt.shape[1], y[:,[0],:])# (1(=batch), pred_len)
        out = out.to('cpu').detach().numpy().copy()

        pred_list = np.concatenate([pred_list, out], axis=0)
        print(iter)
    pred_list = pred_list[1:]
    D = []
    for i in range(predict_step):
        tmp = np.concatenate([all_data[:(input_step+i)], pred_list[:,i], all_data[(all_data.shape[0]-(predict_step - (i + 1))):]])
        D.append(tmp)

    for step_num, tmp_data in enumerate(D):
        n = step_num + 1

        # scatter plot
        corr = scipy.stats.pearsonr(tmp_data, all_data)
        logger({"corr" : corr})

        np.save(os.path.join(scatter_path, "pred{}_epoch{}".format(n, epoch_num)), pred_list)
        np.save(os.path.join(scatter_path, "tgt"), tgt)

        fig, ax = plt.subplots(figsize=(12,8))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.scatter(tmp_data,all_data)
        ax.set_title("{} pred_{} epoch : {}".format(check_state, n, epoch_num))
        ax.set_xlabel("predict")
        ax.set_ylabel("true")
        ax.text(-0.95, -0.5, "corr : {}".format(corr))
        ax.axhline(0)
        ax.axvline(0)
        plt.savefig(os.path.join(scatter_path, "{}_pred{}_scatter_epoch{}.png".format(check_state,n,epoch_num)))
        plt.close()

    # forcast plot
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(all_data, label="true")

    for ind, d in enumerate(D):
        ax.plot(d, label="forcast_{}".format(ind+1))
    ax.legend()
    ax.set_title("forcast plot ({}) epoch {}".format(check_state, epoch_num))
    plt.savefig(os.path.join(scatter_path, "{}  forcast_epoch{}.png".format(check_state, epoch_num)))
    plt.close()


if __name__ == "__main__":
    main()
