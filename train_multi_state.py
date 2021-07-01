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

import util
import model
plt.rcParams["font.size"] = 18

# usege
# zsh ./recipes/run_train.sh

SEED = 42

def make_dataloader(train_data, test_data, input_step, predict_step, use_clm, batch_size, use_state):
    train_set = util.ILIDataset(train_data, input_step, predict_step, use_clm, use_state)
    test_set = util.ILIDataset(test_data, input_step, predict_step, use_clm, use_state)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='Test Voice Transformer Network')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train_data_root', '-train', default="./data/train/", type=str, help='root of train data')
    parser.add_argument('--test_data_root', '-test', default="./data/test/", type=str, help='root of test data')
    parser.add_argument('--outpath', '-out', default="./checkout", type=str, help='root of output directory')
    parser.add_argument('--input_step', '-instep', type=int, help='root of output directory')
    parser.add_argument('--predict_step', '-predstep', type=int, help='root of output directory')
    parser.add_argument('--use_clm', '-uc', type=int, nargs="+", default=0, help='index of column')# usege : -uc 1 2 3 4 (then args.use_clm will be [1,2,3,4])
    parser.add_argument("--d_model", "-d", default=512, type=int, help="model dimention")
    parser.add_argument('--batch_size', '-bs', default=64, type=int, help='batch size')
    parser.add_argument('--max_epoch', '-me', default=1, type=int, help='epoch num')
    parser.add_argument('--N_enc', '-ne', default=4, type=int, help='number of encoder layer')
    parser.add_argument('--N_dec', '-nd', default=4, type=int, help='number of decoder layer')
    parser.add_argument('--h_enc', '-he', default=8, type=int, help='head num of encoder layer')
    parser.add_argument('--h_dec', '-hd', default=8, type=int, help='head num of decoder layer')
    parser.add_argument('--ff_hidnum', '-fh', default=1024, type=int, help='hid num of FeedForward Layer')
    parser.add_argument('--hid_pre', '-hpre', default=1024, type=int, help='hidnum pf PreLayer')
    parser.add_argument('--hid_post', '-hpost', default=1024, type=int, help='hidnum of PostLayer')
    parser.add_argument('--dropout_pre', '-dpre', default=0.2, type=float, help='dropout ratio of PreLayer')
    parser.add_argument('--dropout_post', '-dpost', default=0.2, type=float, help='dropout ratio of PostLayer')
    parser.add_argument('--dropout_model', '-dmodel', default=0.2, type=float, help='dropout ratio of Transformer')
    parser.add_argument('--save_each', '-se', type=int, default=10, help='save each epoch')
    parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='learning ratio')
    parser.add_argument('--debug', '-dbg', action="store_true", help='over write modelfile')
    parser.add_argument('--scheduling', '-sch', type=str, help='type of scheduling', default="No")
    parser.add_argument('--loss_fn', '-loss', type=str, help='type of loss fn', default="mse")
    args = parser.parse_args()

    outpath = args.outpath
    train_data_root = args.train_data_root
    test_data_root = args.test_data_root
    input_step = args.input_step
    predict_step = args.predict_step
    use_clm = args.use_clm
    d_model = args.d_model
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    in_dim = len(use_clm)
    attn_type = "full"
    N_enc = args.N_enc
    N_dec = args.N_dec
    h_enc = args.h_enc
    h_dec = args.h_dec
    ff_hidnum = args.ff_hidnum
    hid_pre = args.hid_pre
    hid_post = args.hid_post
    dropout_pre = args.dropout_pre
    dropout_post = args.dropout_post
    dropout_model = args.dropout_model
    save_each = args.save_each
    lr = args.lr
    use_state = "all"
    dbg = args.debug
    sch = args.scheduling
    # use_clm = [use_clm]
    loss_fn = args.loss_fn

    use_clm_name = util.list2string(use_clm)

    ###################################################
    #############  WARNING and ASSERTION  #############
    ###################################################

    if dropout_pre != 0.0 or dropout_post != 0.0:
        print("WARNING : This model is using simple linear layer for pre and post layer. \n So, pre and post layer's dopout ratio is no meaning.")
    
    assert loss_fn in ["rmse", "mse"], "loss function must be mse or rmse"
 
    #####################################

    device = torch.device("cuda:{}".format(args.gpu)) if args.gpu >= 0 else torch.device("cpu")
    util.fix_seed(SEED)

    model_name = "model_multi_state_instep{}_predstep{}_clmnum{}_d{}_bs{}_maxepc{}_nenc{}_ndec{}_henc{}_hdec{}_ffh{}_prh{}_posth{}_dorpre{}_dopost{}_dormodel{}_lr{}_sch{}_loss{}".format(
        input_step,
        predict_step,
        use_clm_name,
        d_model,
        batch_size,
        max_epoch,
        N_enc,
        N_dec,
        h_enc,
        h_dec,
        ff_hidnum,
        hid_pre,
        hid_post,
        dropout_pre,
        dropout_post,
        dropout_model,
        lr,
        sch,
        loss_fn
    )

    # make directory
    util.set_directories(outpath, model_name, ["plot", "config", "model", "log"],dbg)
    log_path = os.path.join(outpath, model_name, "log")
    model_path = os.path.join(outpath, model_name, "model")
    config_path = os.path.join(outpath, model_name, "config")
    plot_path = os.path.join(outpath, model_name, "plot")

    config = {
            "outpath" : outpath,
            "train_data_root" : train_data_root,
            "test_data_root" : test_data_root,
            "input_step" : input_step,
            "predict_step" : predict_step,
            "use_clm" : use_clm,
            "d_model" : d_model,
            "batch_size" : batch_size,
            "max_epoch" : max_epoch,
            "attn_type" : "full",
            "N_enc" : N_enc,
            "N_dec" : N_dec,
            "h_enc" : h_enc,
            "h_dec" : h_dec,
            "ff_hidnum" : ff_hidnum,
            "hid_pre" : hid_pre,
            "hid_post" : hid_post,
            "dropout_pre" : dropout_pre,
            "dropout_post" : dropout_post,
            "dropout_model" : dropout_model,
            "save_each" : save_each,
            "use_state" : use_state,
            "sch" : sch,
            "loss_fn" : loss_fn
    }
    util.save_config(config_path, config)

    logger = util.Logger(log_path, "log", args_dict=config)

    train_data, test_data = util.load_data(train_data_root, test_data_root)

    # make dataloader
    train_loader, test_loader = make_dataloader(train_data, test_data, input_step, predict_step, use_clm, batch_size, use_state)

    # train
    train_loss_log, val_loss_log, max_iteration = train(device, train_loader, test_loader, logger, d_model, in_dim, attn_type, N_enc, N_dec, h_enc, h_dec, ff_hidnum, hid_pre, hid_post, dropout_pre, dropout_post, dropout_model, max_epoch, save_each, model_path, lr, sch, loss_fn, plot_path)
    plot_loss(plot_path, train_loss_log, val_loss_log, max_iteration)
    
    print(model_name)


def train(device, train_loader, test_loader, logger, d_model, in_dim, attn_type, N_enc, N_dec, h_enc, h_dec, ff_hidnum, hid_pre, hid_post, dropout_pre, dropout_post, dropout_model, max_epoch, save_each, model_path, lr, sch, loss_fn, plot_root):
    net = model.Transformer(device, d_model, in_dim, attn_type, N_enc, N_dec, h_enc, h_dec, ff_hidnum, hid_pre, hid_post, dropout_pre, dropout_post, dropout_model)
    net = net.to(device)
    
    train_loss_log = []
    val_loss_log = []

    if sch == "normal":
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.999))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90) # from VTN
    elif sch == "lambda":
        optimizer = optim.Adam(net.parameters(), lr=1.0, betas=(0.9,0.98))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=partial(util.lr_func, d_model = d_model))# from 論文
    elif sch == "No":
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.98))
    else:
        assert False, "please set correct scheduler"
    
    criterion = nn.MSELoss()
    max_iteration = 0
    for epoch in range(max_epoch):
        # 学習結果もscatter_plotできるようにする。
        tgt_log_train = np.zeros((1,1))
        pred_log_train = np.zeros((1,1))

        for iter, (x, y, tgt, key) in enumerate(train_loader):
            x, y, tgt = x.to(device), y.to(device), tgt.to(device)
            tgt = tgt[:,:,0]
            net.train()
            
            out = net(x, y)
            loss = criterion(out, tgt)

            if loss_fn == "rmse":
                loss = torch.sqrt(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not sch == "No":
                scheduler.step()

            train_loss_log.append(loss.item())
            
            print("epoch : {}, itr :{}, loss : {}, RMSE : {}".format(epoch, iter, loss.item(), torch.sqrt(loss).item()))
            msg = {
                "epoch" : epoch,
                "iteration" : iter,
                "loss" : loss.item(),
                "RMSE" : torch.sqrt(loss).item()
            }

            logger(msg)
            
            out_npy = out.to('cpu').detach().numpy().copy()
            tgt_npy = tgt.to('cpu').detach().numpy().copy()

            pred_log_train = np.concatenate([pred_log_train, out_npy[:,[0]]], axis=0)
            tgt_log_train = np.concatenate([tgt_log_train, tgt_npy[:,[0]]], axis=0)
        
        if epoch == 0:
            max_iteration = iter + 1
        
        if epoch % save_each == 0 or epoch == max_epoch - 1:
            scatter_plot(pred_log_train, tgt_log_train, plot_root, epoch, "train", logger)

        val_loss = check_data(device, test_loader, net, criterion)
        val_loss_log.append(val_loss)# MSE

        print("*************validation******************")
        print("val_loss : {}, rmse : {}".format(val_loss, math.sqrt(val_loss)))

        msg = {
            "validation" : "*************************************************** \r\n",
            "epoch" : epoch,
            "loss" : val_loss,
            "rmse" : math.sqrt(val_loss)
        }
        logger(msg)

        # save model
        if epoch % save_each == 0 or epoch == max_epoch - 1:
            torch.save(net.state_dict(), os.path.join(model_path, '{}_epoch.model'.format(epoch)))
            check_scatter(device, test_loader, net, plot_root, epoch, "validation", logger)
        
    return train_loss_log, val_loss_log, max_iteration

def check_data(device, test_loader, net, criterion):
    net.eval()
    loss_log = []
    for iter, (x, y, tgt, key) in enumerate(test_loader):
        x, y, tgt = x.to(device), y.to(device), tgt.to(device)
        tgt = tgt.squeeze(-1)
        net.train()
        
        out = net(x, y)

        loss = criterion(out, tgt)
        loss_log.append(loss.item())
    
    return sum(loss_log)/len(loss_log)# MSE

def check_scatter(device, test_loader, net, plot_path, epoch, train_test, logger=None):
    net.eval()
    tgt_log = np.zeros((1,1))
    pred_log = np.zeros((1,1))
    for iter, (x, y, tgt, key) in enumerate(test_loader):
        x, y, tgt = x.to(device), y.to(device), tgt.to(device)
        tgt = tgt.squeeze(-1)
        net.train()
        
        out = net(x, y)
        # out = F.sigmoid(out)

        out_npy = out.to('cpu').detach().numpy().copy()
        tgt_npy = tgt.to('cpu').detach().numpy().copy()

        pred_log = np.concatenate([pred_log, out_npy[:,[0]]], axis=0)
        tgt_log = np.concatenate([tgt_log, tgt_npy[:,[0]]], axis=0)
    
    scatter_plot(pred_log, tgt_log, plot_path, epoch, train_test, logger)

def scatter_plot(pred, tgt, plot_path, epoch, train_test, logger=None):
    # culc corr
    pred = pred[:,0]
    tgt = tgt[:,0]
    corr = scipy.stats.pearsonr(pred,tgt)

    np.save(os.path.join(plot_path, "pred_epoch{}_{}".format(epoch, train_test)), pred)
    np.save(os.path.join(plot_path, "tgt_epoch{}_{}".format(epoch, train_test)), tgt)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.scatter(pred,tgt)
    ax.set_title("epoch : {}  {}".format(epoch, train_test))
    ax.text(-0.95, -0.5, "corr : {}".format(corr))
    ax.axhline(0)
    ax.axvline(0)
    plt.savefig(os.path.join(plot_path, "sct_epoch{}_{}.png".format(epoch, train_test)))
    plt.close()
    print(f"{train_test}  epoch : {epoch}, corr : {corr}")
    msg = {
        "train_test" : train_test,
        "epoch" : epoch,
        "corr" : corr
    }
    logger(msg)


def plot_loss(save_path, train_loss_log, val_loss_log, max_iteration):
    train_loss_log = np.array(train_loss_log)
    val_loss_log = np.array(val_loss_log)

    plot_path =  os.path.join(save_path, "loss.png")
    np_path_train = os.path.join(save_path, "loss_train.np")
    np_path_val = os.path.join(save_path, "loss_val.np")

    x_val = np.linspace(max_iteration, max_iteration*len(val_loss_log), len(val_loss_log)) - 1

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(train_loss_log, label="train")
    ax.plot(x_val, val_loss_log, label="validation")
    ax.set_title("loss")
    plt.legend()
    plt.savefig(plot_path)
    plt.clf()
    plt.close()

    np.save(np_path_train, train_loss_log)
    np.save(np_path_val, val_loss_log)

if __name__ == "__main__":
    main()
