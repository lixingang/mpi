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

# import in-project packages
from config import train_config
from Losses.loss import HEMLoss,CenterLoss
from Models.network import Net
from Models.mpi_datasets import mpi_dataset
sys.path.append("./Metrics")
import Metrics.torchmetrics as torchmetrics
from Utils.AverageMeter import AverageMeter
from Utils.clock import clock,Timer
from Utils.setup_seed import setup_seed

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--other', default=None,)
for key in train_config:
    parser.add_argument(f'--{key}', default=train_config[key],)
# parser.add_argument(f'--tag', default="",)
args = parser.parse_args()

def logging_setting(args):
    logging.basicConfig(
        # filename='new.log', filemode='w',
        format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        # format='%(asctime)s  %(levelname)-10s \033[0;33m%(message)s\033[0m',
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.INFO 
    )

def predict(args):
    ds = ConcatDataset([mpi_dataset(args, h5path) for h5path in args.h5_dir])
    train_size = int(len(ds) * 0.6)
    validate_size = int(len(ds) * 0.2)
    test_size = len(ds) - validate_size - train_size
    logging.info(f"train,validate,test size: {train_size},{validate_size},{test_size}")

    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, validate_size, test_size])
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1,
        drop_last=False,
    )
    device = torch.device(args.device)
    model = Net(args).to(device)
    metrics = {
        "r2": torchmetrics.R2Score().to(device),
        "mape": torchmetrics.MeanAbsolutePercentageError().to(device),
        "mse": torchmetrics.MeanSquaredError().to(device),
    }
    model.eval()
    if args.best_weight is not None:
        model.load_state_dict(torch.load(args.best_weight))
    _ = [metrics[k].reset() for k in metrics.keys()]
    res = {"y":[],"y_hat":[]}
    for y, fea_img, fea_num in test_loader:
        y = y.cuda()
        y_hat = model(fea_img.cuda(), fea_num.cuda())
        acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
        res["y"].extend(y.squeeze().tolist())
        res["y_hat"].extend(y_hat.squeeze().tolist())

    logging.info(f"[test] Testing with {args.best_weight}")
    logging.info(f"[test] r2={acc['r2']:.3f} rmse={acc['mse']:.3f} mape:{acc['mape']:.3f}")
    return res
    
if __name__=='__main__':
    
    logging_setting(args)
    setup_seed(args.seed)
    args.best_weight = "Logs/Mar16_16-51-23/mpi_epoch80_r2_0.7185.pth"
    args.log_dir = os.path.dirname(args.best_weight)
    print(args.log_dir)
    res = predict(args)
    y,y_hat = res['y'],res['y_hat']
    sorted_id = sorted(range(len(y)), key=lambda k: y[k])
    y = [y[i] for i in sorted_id]
    y_hat = [y_hat[i] for i in sorted_id]

    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax1 = fig.add_subplot(211)
    ax1.set_xlabel('The mpi_fixed3'); ax1.set_xlim([-0.2, 0.8])
    ax1.set_ylabel('volts')
    ax1.hist(y, bins=100,rwidth=0.8)

    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('The predicted MPI '); ax2.set_xlim([-0.2, 0.8])
    ax2.set_ylabel('volts')
    ax2.hist(y_hat, bins=100,rwidth=0.8)

    fig.tight_layout()
    plt.savefig(os.path.join(args.log_dir,'test.png'))