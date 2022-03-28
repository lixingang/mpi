from distutils.log import error
import matplotlib.pyplot as plt
import os,sys
import torch
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim.lr_scheduler import StepLR,LambdaLR,MultiStepLR
import logging
import argparse
import h5py
import glob
import time,random,datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# import in-project packages
from Losses.loss import HEMLoss,CenterLoss
from Models.network import Net
from Models.mpi_datasets import mpi_dataset
sys.path.append("./Metrics")
import Metrics.torchmetrics as torchmetrics
from Utils.AverageMeter import AverageMeter
from Utils.clock import clock,Timer
from Utils.setup_seed import setup_seed
from Utils.ParseYAML import ParseYAML
import math
def logging_setting(args):
    logging.basicConfig(
        # filename='new.log', filemode='w',
        format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        # format='%(asctime)s  %(levelname)-10s \033[0;33m%(message)s\033[0m',
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.INFO 
    )
def get_bin_index(y, y_hat, min_value, max_value, num_bin):
    error = [y_hat[i]-y[i] for i in range(len(y))]
    unit = (max_value-min_value)/num_bin
    xbins = np.arange(min_value, max_value, unit, dtype=float )
    error_list = [0 for i in range(num_bin)]
    count_list = [0 for i in range(num_bin)]
    for i in range(len(y)):
        index = int(y[i]//unit)
        index = index if index<num_bin-1 else num_bin-1
        
        error_list[index]+=abs(error[i])
        count_list[index]+=1
    error_list = [error_list[i]/count_list[i] if count_list[i]!=0 else 0 for i in range(len(error_list))]
    return xbins, error_list, count_list, 

def predict(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    ds = ConcatDataset([mpi_dataset(args, h5path) for h5path in args.h5_dir])
    train_size = int(len(ds) * 0.7)
    validate_size = int(len(ds) * 0.15)
    test_size = len(ds) - validate_size - train_size
    logging.info(f"train,validate,test size: {train_size},{validate_size},{test_size}")

    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, validate_size, test_size])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    validate_loader = DataLoader(
        validate_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1, 
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1,
        drop_last=False,
    )
    model = Net(args).cuda()
    metrics = {
        "r2": torchmetrics.R2Score().cuda(),
        "mape": torchmetrics.MeanAbsolutePercentageError().cuda(),
        "mse": torchmetrics.MeanSquaredError().cuda(),
    }
    with torch.no_grad():
        assert args.best_weight_path is not None
        model.load_state_dict(torch.load(args.best_weight_path))
        _ = [metrics[k].reset() for k in metrics.keys()]
        train_res = {"y":[],"y_hat":[]}
        validate_res = {"y":[],"y_hat":[]}
        test_res = {"y":[],"y_hat":[]}
        for y, fea_img, fea_num in train_loader:
            y = y.cuda()
            y_hat = model(fea_img.cuda(), fea_num.cuda())
            acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
            train_res["y"].extend(y.squeeze().tolist())
            train_res["y_hat"].extend(y_hat.squeeze().tolist())

        for y, fea_img, fea_num in validate_loader:
            y = y.cuda()
            y_hat = model(fea_img.cuda(), fea_num.cuda())
            acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
            validate_res["y"].extend(y.squeeze().tolist())
            validate_res["y_hat"].extend(y_hat.squeeze().tolist())

        for y, fea_img, fea_num in test_loader:
            y = y.cuda()
            y_hat = model(fea_img.cuda(), fea_num.cuda())
            acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
            test_res["y"].extend(y.squeeze().tolist())
            test_res["y_hat"].extend(y_hat.squeeze().tolist())

        logging.info(f"[test] Testing with {args.best_weight_path}")
        logging.info(f"[test] r2={acc['r2']:.3f} rmse={acc['mse']:.3f} mape:{acc['mape']:.3f}")
        return train_res, validate_res, test_res 

def plot(args,y,y_hat,savename):
    sorted_id = sorted(range(len(y)), key=lambda k: y[k])
    y = [y[i] for i in sorted_id]
    y_hat = [y_hat[i] for i in sorted_id]
    n_bin = 100
    fig = plt.figure(figsize=(10, 6), dpi=200)
    ax1 = fig.add_subplot(311)
    ax1.set_xlabel('The mpi_fixed3'); ax1.set_xlim([-0.8, 0.8])
    ax1.set_ylabel('volts')
    n, bins, patches = ax1.hist(y, bins=100,rwidth=0.8)

    ax2 = fig.add_subplot(312)
    ax2.set_xlabel('The predicted MPI '); ax2.set_xlim([-0.8, 0.8])
    ax2.set_ylabel('volts')
    n, bins, patches = ax2.hist(y_hat, bins=100,rwidth=0.8)

    ax3 = fig.add_subplot(313)
    ax3.set_xlabel('The predicted MPI ');
    ax3.set_ylabel('volts')
    n, bins, patches = ax3.hist([y_hat[i]-y[i] for i in range(len(y))], bins=100,rwidth=0.8)

    fig.tight_layout()
    plt.savefig(os.path.join(args.log_dir, savename))

def plot_list(args, xbins, y, savename):
    fig = plt.figure(figsize=(10, 6), dpi=200)
    plt.bar(xbins, y)
    plt.bar(xbins, y)
    plt.savefig(os.path.join(args.log_dir, savename))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(f'--log_dir', default= "Logs/Mar20_20-43-30",)
    args = parser.parse_args()
    args = ConfigDict(os.path.join(args.log_dir, "config.yaml"))
    logging_setting(args)
    setup_seed(args.seed)
    train_res, valid_res, test_res = predict(args)
    # plot(args, train_res['y'],train_res['y_hat'], "train_res.png")
    # plot(args, valid_res['y'],valid_res['y_hat'], "valid_res.png")
    # plot(args, test_res['y'],test_res['y_hat'], "test_res.png")
    # plot()
    y = train_res['y']
    y_hat = train_res['y_hat']
    print("Train Pearsonr(y,y_hat):", pearsonr(y,y_hat))
    xbins, error_list, count_list  = get_bin_index(y,y_hat,0,1,100)
    # print(xbins, error_list, count_list, )
    plot_list(args, range(len(error_list)), error_list, "train_error_list.png")
    plot_list(args,  range(len(count_list)), count_list, "train_count_list.png")

    y = test_res['y']
    y_hat = test_res['y_hat']
    print("Test Pearsonr(y,y_hat):", pearsonr(y,y_hat))
    xbins, error_list, count_list  = get_bin_index(y,y_hat,0,1,100)
    # print(xbins, error_list, count_list, )
    plot_list(args, range(len(error_list)), error_list, "test_error_list.png")
    plot_list(args,  range(len(count_list)), count_list, "test_count_list.png")

    