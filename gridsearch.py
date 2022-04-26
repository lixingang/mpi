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
from Models.gp import gp_model
from Datasets.mpi_datasets import mpi_dataset
sys.path.append("./Metrics")

from Utils.AverageMeter import AverageMeter
from Utils.clock import clock,Timer
from Utils.base import setup_seed
from Utils.parse import ParseYAML,parse_log

def logging_setting():
    logging.basicConfig(
        # filename='new.log', filemode='w',
        format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        # format='%(asctime)s  %(levelname)-10s \033[0;33m%(message)s\033[0m',
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.INFO 
    )


def _test_epoch(args, epoch, loader, gp=None, writer=None):
    with torch.no_grad():
        test_model = Net(args).cuda()
        test_model.eval()
        # restore the parameters
        test_model.load_state_dict(torch.load(args.best_weight_path))
        if gp:
            gp.restore(args.best_gp_path)
        y = []
        y_hat = []
        names = []
        lons = []
        lats = []
        for fea, lbl in loader:
            img_data = fea[0]
            num_data = fea[1]
            
            _y = lbl["MPI3_fixed"].cuda()
            _y_hat, fea = test_model(img_data.cuda(), num_data.cuda())
            # print(_y_hat[0])
            if gp:
                gp.append_testing_params(
                    fea.detach().cpu().numpy(), 
                    lbl['year'], 
                    np.stack([lbl['lat'], lbl['lon']],-1),
                )
            y.append(_y)
            y_hat.append(_y_hat)
            names.extend(lbl['name'])
            lons.extend(lbl['lon'].tolist())
            lats.extend(lbl['lat'].tolist())
        y = torch.cat(y, dim=0).detach()
        y_hat = torch.cat(y_hat, dim=0).detach() 

        if gp:
            y_hat = gp.gp_run(
                epoch,
                test_model.state_dict()["fclayer.3.weight"].cpu(),
                test_model.state_dict()["fclayer.3.bias"].cpu(),
            ).cuda()
        r2 = torchmetrics.functional.r2_score(y_hat, y).cpu().numpy().item()
        mse = torchmetrics.functional.mean_squared_error(y_hat, y).cpu().numpy().item()
        acc = {"test/r2":r2, "test/mse":mse}

        logging.info(f"[test] Testing with {args.best_weight_path}")
        logging.info(f"[test] r2={r2:.3f} mse={mse:.4f}")

        df = pd.DataFrame({"name":names, "y":y.cpu().tolist(), "y_hat":y_hat.cpu().tolist(), "lon":lons, "lat":lats,})
        df.to_csv(os.path.join(args.log_dir,"predict.csv"))
        
        # model.load_state_dict(training_weight)

        return acc 


if __name__=='__main__':
    logging_setting()
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(f'--name', default= "Logs/Mar20_20-43-30",)
    args = parser.parse_args()
    name = args.name
    gp = gp_model(sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01)
    for logdir in glob.glob(f"Logs/{name}/*"):
        if os.path.isfile(logdir):
            continue
        args = ParseYAML(os.path.join(logdir, "config.yaml"))
        setup_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
        data_list = ParseYAML(os.path.join(args.log_dir, "train_valid_test.yaml"))
        # train_dataset = mpi_dataset(args, data_list["train_list"])
        # valid_dataset = mpi_dataset(args, data_list["valid_list"])
        # test_dataset = mpi_dataset(args, data_list["test_list"])
        pred_dataloader = DataLoader(
            mpi_dataset(args, data_list["test_list"]), 
            batch_size=2, 
            shuffle=False, 
            num_workers=0,
            drop_last=False,
        )

        model = Net(args).cuda()
        metrics = {
            "r2": torchmetrics.R2Score().cuda(),
            "mse": torchmetrics.MeanSquaredError().cuda(),
        }
        
        acc = _test_epoch(args, 1, pred_dataloader, None, None)
        # print(name, acc["test/r2"],acc["test/mse"])

   