from Utils.parse import ParseYAML, parse_log
from Utils.clock import clock, Timer
from Utils.base import setup_seed
from Utils.AverageMeter import AverageMeter
from distutils.log import error
import matplotlib.pyplot as plt
import math
import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR
import logging
import argparse
import h5py
import glob
import time
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torchmetrics
import pandas as pd

# import in-project packages
from Losses.loss import HEMLoss, CenterLoss
from Models import *
from Datasets.mpi_datasets import mpi_dataset
sys.path.append("./Metrics")


def logging_setting(args):
    logging.basicConfig(
        # filename='new.log', filemode='w',
        format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        # format='%(asctime)s  %(levelname)-10s \033[0;33m%(message)s\033[0m',
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.INFO
    )


def get_hist(y, min_value=0.0, maxs_value=1., step=0.01):
    sorted_id = sorted(range(len(y)), key=lambda k: y[k])
    y = np.array([y[i] for i in sorted_id])
    hist_index = {}
    iter_list = np.arange(min_value, maxs_value, step)
    for i in iter_list:
        start = i
        end = i+step
        hist_index[round(end, 2)] = np.where((y > start) & (y <= end))[0]
    return hist_index, iter_list


def get_model(args):
    assert args.model.lower() in {'vit', 'mlp'}
    model = None
    if args.model.lower() == "vit":
        model = ViT(
            num_patches=len(args.img_keys)+len(args.num_keys),
            patch_dim=args.in_channel,
            num_classes=1,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim,
            dropout=args.vit_dropout,
            emb_dropout=args.vit_dropout
        ).cuda()
    elif args.model.lower() == 'mlp':
        model = MLP(args).cuda()

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(f'--log_dir', default="Logs/Mar20_20-43-30",)
    args = parser.parse_args()
    args = ParseYAML(os.path.join(args.log_dir, "config.yaml"))
    logging_setting(args)
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data_list = ParseYAML(os.path.join(args.log_dir, "train_valid_test.yaml"))
    train_dataset = mpi_dataset(args, data_list["train_list"])
    valid_dataset = mpi_dataset(args, data_list["valid_list"])
    test_dataset = mpi_dataset(args, data_list["test_list"])

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    with torch.no_grad():
        test_model = get_model(args)
        test_model.eval()
        assert args.best_weight_path is not None
        test_model.load_state_dict(torch.load(args.best_weight_path))
        if args.use_gp:
            gp = gp_model(sigma=1, r_loc=2.5, r_year=3.,
                          sigma_e=0.32, sigma_b=0.01)
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
            if args.use_gp:
                gp.append_testing_params(
                    fea.detach().cpu().numpy(),
                    lbl['year'],
                    np.stack([lbl['lat'], lbl['lon']], -1),
                )
            y.append(_y)
            y_hat.append(_y_hat)
            names.extend(lbl['name'])
            lons.extend(lbl['lon'].tolist())
            lats.extend(lbl['lat'].tolist())
        y = torch.cat(y, dim=0).detach()
        y_hat = torch.cat(y_hat, dim=0).detach()
        if args.use_gp:
            y_hat = gp.gp_run(
                test_model.state_dict()["fclayer.3.weight"].cpu(),
                test_model.state_dict()["fclayer.3.bias"].cpu(),
            ).cuda()
        r2 = torchmetrics.functional.r2_score(y_hat, y).cpu().numpy().item()
        mse = torchmetrics.functional.mean_squared_error(
            y_hat, y).cpu().numpy().item()
        acc = {"test/r2": r2, "test/mse": mse}

        logging.info(f"[test] Testing with {args.best_weight_path}")
        logging.info(f"[test] r2={r2:.3f} mse={mse:.4f}")
        res = {"name": names, "y": y.cpu().tolist(
        ), "y_hat": y_hat.cpu().tolist(), "lon": lons, "lat": lats, }
        df = pd.DataFrame(res)
        df.to_csv(os.path.join(args.log_dir, "predict.csv"))

    y = np.array(res['y'])
    y_hat = np.array(res['y_hat'])
    diff = np.abs(y-y_hat)

    hist_index, iter_list = get_hist(y, 0., 1., 0.02)
    hist_count = [len(hist_index[k]) for k in hist_index.keys()]
    # TODO 有问题！
    hist_error = []
    for key in hist_index.keys():
        if len(hist_index[key]) != 0:
            hist_error.append(np.average(diff[hist_index[key]]))
        else:
            hist_error.append(0)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    axs[0, 0].bar(iter_list, hist_count, width=0.01)
    axs[0, 0].set_xlabel('The count distribution')

    axs[0, 1].bar(iter_list, hist_error, width=0.01)
    axs[0, 1].set_xlabel('The error distribution')

    axs[1, 0].hist(y, bins=100, rwidth=0.8)
    axs[1, 0].set_xlabel('y')
    # axs[1,0].set_xlim([-0.8, 0.8])

    axs[1, 1].hist(y_hat, bins=100, rwidth=0.8)
    axs[1, 1].set_xlabel('y_hat')
    # axs[1,1].set_xlim([-0.8, 0.8])
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, "vis_results.png"))

    indexs = list(range(len(hist_count)))
    nonzero_index = [i for i in indexs if hist_count[i] > 0]
    hist_count = np.array(hist_count)[nonzero_index]
    hist_error = np.array(hist_error)[nonzero_index]
    # print(hist_count, hist_error)
    print("Pearsonr(y,y_hat):", pearsonr(hist_count, hist_error))
    # xbins, error_list, count_list  = get_bin_index(y,y_hat,0,1,100)
    # print(xbins, error_list, count_list, )
    # plot_list(args, range(len(error_list)), error_list, "error_list.png")
    # plot_list(args,  range(len(count_list)), count_list, "count_list.png")
