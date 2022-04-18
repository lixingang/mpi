from distutils.log import error
import matplotlib.pyplot as plt
import math
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
import torchmetrics 
import pandas as pd

# import in-project packages
from Losses.loss import HEMLoss,CenterLoss
from Models.network import Net
from Datasets.mpi_datasets import mpi_dataset
sys.path.append("./Metrics")

from Utils.AverageMeter import AverageMeter
from Utils.clock import clock,Timer
from Utils.setup_seed import setup_seed
from Utils.ParseYAML import ParseYAML

def logging_setting(args):
    logging.basicConfig(
        # filename='new.log', filemode='w',
        format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        # format='%(asctime)s  %(levelname)-10s \033[0;33m%(message)s\033[0m',
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.INFO 
    )

def get_hist(y, min_value=0.0, maxs_value=1., step=0.01 ):
    sorted_id = sorted(range(len(y)), key=lambda k: y[k])
    y = np.array([y[i] for i in sorted_id])
    hist_index = {}
    iter_list = np.arange(min_value, maxs_value, step)
    for i in iter_list:
        start = i
        end = i+step
        hist_index[round(end,2)] = np.where((y>start)&(y<=end))[0]
    return hist_index, iter_list
    

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(f'--log_dir', default= "Logs/Mar20_20-43-30",)
    args = parser.parse_args()
    args = ParseYAML(os.path.join(args.log_dir, "config.yaml"))
    logging_setting(args)
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    data_list = ParseYAML(os.path.join(args.log_dir, "train_valid_test.yaml"))
    train_dataset = mpi_dataset(args, data_list["train_list"])
    valid_dataset = mpi_dataset(args, data_list["valid_list"])
    test_dataset = mpi_dataset(args, data_list["test_list"])
    

    any_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,
        pin_memory=True,
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
        res = {"name":[],"y":[],"y_hat":[]}
        for fea, lbl in any_dataloader:
            fea_img = fea[0]
            fea_num = fea[1]
            y = lbl["MPI3_fixed"].cuda()
            y_hat = model(fea_img.cuda(), fea_num.cuda())
            acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
            res["name"].extend(lbl["name"])
            res["y"].extend(y.cpu().numpy())
            res["y_hat"].extend(y_hat.cpu().numpy())
        acc = {k: metrics[k].compute() for k in metrics.keys()}
        logging.info(f"Testing with {args.best_weight_path}")
        logging.info(f"r2={acc['r2']:.3f} rmse={acc['mse']:.3f} mape:{acc['mape']:.3f}")
        res = pd.DataFrame(res)
        res.to_csv(f"{args.log_dir}/test_result.csv") 
        
    y = np.array(res["y"])
    y_hat = np.array(res['y_hat'])
    diff = np.abs(y-y_hat)
    
    hist_index, iter_list = get_hist(y,0.,1.,0.01)
    hist_count = [len(hist_index[k]) for k in hist_index.keys()]
    hist_error = [np.average(diff[hist_index[key]]) if len(hist_index[key])!=0 else 0 for key in hist_index.keys()]

    
    fig, axs = plt.subplots(ncols=2, nrows=2,figsize=(12,10))
    axs[0,0].bar(iter_list, hist_count, width=0.01)
    axs[0,0].set_xlabel('The count distribution')

    axs[0,1].bar(iter_list, hist_error, width=0.01)
    axs[0,1].set_xlabel('The error distribution')
    
    axs[1,0].hist(y, bins=100, rwidth=0.8)
    axs[1,0].set_xlabel('y') 
    # axs[1,0].set_xlim([-0.8, 0.8])

    axs[1,1].hist(y_hat, bins=100, rwidth=0.8)
    axs[1,1].set_xlabel('y_hat')
    # axs[1,1].set_xlim([-0.8, 0.8])
    plt.tight_layout() 
    plt.savefig(os.path.join(args.log_dir, "vis_results.png"))


    print("Pearsonr(y,y_hat):", pearsonr(hist_count,hist_error))
    # xbins, error_list, count_list  = get_bin_index(y,y_hat,0,1,100)
    # print(xbins, error_list, count_list, )
    # plot_list(args, range(len(error_list)), error_list, "error_list.png")
    # plot_list(args,  range(len(count_list)), count_list, "count_list.png")


   