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
        level=logging.DEBUG 
    )
def save_args(args):
    argsDict = args.__dict__
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'setting.txt', 'w')) as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

def predict(args):
    ds = ConcatDataset([mpi_dataset(args, h5path) for h5path in args.h5_dir])
    train_size = int(len(ds) * 0.6)
    validate_size = int(len(ds) * 0.2)
    test_size = len(ds) - validate_size - train_size
    logging.info(f"train,validate,test size: {train_size},{validate_size},{test_size}")

    _, _, test_dataset = torch.utils.data.random_split(ds, [train_size, validate_size, test_size])
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=False,
    )
    device = torch.device(args.device)
    model = Net(args).to(device)

    if args.restore_weight is not None:
        model.load_state_dict(torch.load(args.restore_weight))
    
    criterion = HEMLoss(0)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.init_lr, )
    
    metrics = {
        "r2": torchmetrics.R2Score().to(device),
        "mape": torchmetrics.MeanAbsolutePercentageError().to(device),
        "mse": torchmetrics.MeanSquaredError().to(device),
    }
    model.eval()
    if args.best_weight is not None:
        model.load_state_dict(torch.load(args.best_weight))
    _ = [metrics[k].reset() for k in metrics.keys()]
    losses = AverageMeter()
    for y, fea_img, fea_num in test_loader:
        y = y.cuda()
        y_hat = model(fea_img.cuda(), fea_num.cuda())
        loss = criterion(y_hat, y)
        losses.update(loss)
        acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
    
    logging.info(f"[test] Testing with {args.best_weight}")
    logging.info(f"[test] r2={acc['r2']:.3f} rmse={acc['mse']:.3f} mape:{acc['mape']:.3f}")

if __name__=='__main__':
    logging_setting(args)
    setup_seed(args.seed)
    print(predict(args))
