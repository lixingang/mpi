# import office packages
import os,sys
import torch
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim.lr_scheduler import StepLR,LambdaLR
from torch.utils.tensorboard import SummaryWriter
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
args = parser.parse_args()

setup_seed(args.seed)

def logging_setting(args):
    filename = os.path.join(args.log_dir, f"{args.model_name}.txt")
    logging.basicConfig(
        # filename='new.log', filemode='w',
        format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        # format='%(asctime)s  %(levelname)-10s \033[0;33m%(message)s\033[0m',
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.DEBUG 
    )


def train(args):
    logging_setting(args)
    writer = SummaryWriter(log_dir=args.log_dir)
    logging.info(f"The Dataset: {[h5path for h5path in args.h5_dir]}")
    ds = ConcatDataset([mpi_dataset(args, h5path) for h5path in args.h5_dir])

    train_size = int(len(ds) * 0.6)
    validate_size = int(len(ds) * 0.2)
    test_size = len(ds) - validate_size - train_size
    logging.info(f"train,validate,test size: {train_size},{validate_size},{test_size}")

    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, validate_size, test_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=10,
        pin_memory=True,
        drop_last=False,
    )
    validate_loader = DataLoader(
        validate_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=5, 
        pin_memory=True,
        drop_last=False,
    )

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
    optimizer = torch.optim.Adam(model.parameters(),lr=args.init_lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    metrics = {
        "r2": torchmetrics.R2Score().to(device),
        "mape": torchmetrics.MeanAbsolutePercentageError().to(device),
        "mse": torchmetrics.MeanSquaredError().to(device),
    }
    
    for epoch in range(1, args.epochs+1):
        model.train()
        _ = [metrics[k].reset() for k in metrics.keys()]
        losses = AverageMeter()
        for y, fea_img, fea_num in train_loader:
            y = y.cuda()
            y_hat = model(fea_img.cuda(), fea_num.cuda())
            # print(y,y_hat)
            loss = criterion(y_hat, y)
            loss.backward()
            losses.update(loss)
            optimizer.step()
            optimizer.zero_grad()
            acc = {k: metrics[k](y_hat, y) for k in metrics.keys()}
        acc = {k: metrics[k].compute() for k in metrics.keys()}
        writer.add_scalar("Train/loss", losses.avg(), epoch)
        writer.add_scalar("Train/r2", acc['r2'], epoch)
        writer.add_scalar("Train/mse", acc['mse'], epoch)
        logging.info(f"[train] epoch {epoch}/{args.epochs} r2={acc['r2']:.3f} rmse={acc['mse']:.3f} mape:{acc['mape']:.3f}")

        if epoch%5==0:
            model.eval()
            _ = [metrics[k].reset() for k in metrics.keys()]
            losses = AverageMeter()
            for y, fea_img, fea_num in validate_loader:
                y = y.cuda()
                y_hat = model(fea_img.cuda(), fea_num.cuda())
                print(y,y_hat)
                loss = criterion(y_hat, y)
                losses.update(loss)
                acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
            if acc[args.best_acc["name"]]>args.best_acc["value"]:
                args.best_acc["value"] = acc[args.best_acc["name"]]
                os.mkdir(args.log_dir) if not os.path.exists(args.log_dir) else None   
                filename= f"{args.model_name}_epoch{epoch}_{args.best_acc['name']}_{args.best_acc['value']:.4f}.pth"
                args.best_weight = os.path.join(args.log_dir, filename)
                torch.save(model.state_dict(), args.best_weight)
            writer.add_scalar("Validate/loss", losses.avg(), epoch)
            writer.add_scalar("Validate/r2", acc['r2'], epoch)
            writer.add_scalar("Validate/mse", acc['mse'], epoch)
            logging.info(f"[valid] epoch {epoch}/{args.epochs} r2={acc['r2']:.3f} rmse={acc['mse']:.3f} mape:{acc['mape']:.3f}")
            logging.debug(f"epoch{epoch}, Current learning rate: {scheduler.get_last_lr()}")
            
        scheduler.step()
        
        
        if epoch%10==0:
            model.eval()
            training_weight = model.state_dict()
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
            writer.add_scalar("Test/loss", losses.avg(), epoch)
            writer.add_scalar("Test/r2", acc['r2'], epoch)
            writer.add_scalar("Test/mse", acc['mse'], epoch)
            logging.info(f"Testing with {args.best_weight}")
            logging.info(f"[test] r2={acc['r2']:.3f} rmse={acc['mse']:.3f} mape:{acc['mape']:.3f}")
            model.load_state_dict(training_weight)
        
    return "OK"



print(train(args))
