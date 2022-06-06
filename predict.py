from Utils.base import setup_seed
from Utils.base import Meter, parse_yaml, parse_log
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
import fire
from torchmetrics.functional import r2_score, mean_squared_error
from torch.utils.tensorboard import SummaryWriter

# import in-project packages
from Losses.loss import HEMLoss, CenterLoss
from Models import *
from Datasets.mpi_datasets import mpi_dataset
from main import get_model
from Utils.base import parse_yaml

sys.path.append("./Metrics")


def logging_setting():
    logging.basicConfig(
        # filename='new.log', filemode='w',
        format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        # format='%(asctime)s  %(levelname)-10s \033[0;33m%(message)s\033[0m',
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.INFO,
    )


def count_analysis(df, cv_index, writer):
    y = np.array(df["y"])
    y_hat = np.array(df["yhat"])
    diff = np.abs(y - y_hat)
    hist_index, iter_list = get_hist(y, 0.0, 1.0, 0.05)
    hist_count = [len(hist_index[k]) for k in hist_index.keys()]
    hist_error = []
    for key in hist_index.keys():
        if len(hist_index[key]) != 0:
            hist_error.append(np.average(diff[hist_index[key]]))
        else:
            hist_error.append(0)
            # raise RuntimeError("encounter zero sample in the bins")

    f = plt.figure(dpi=200, figsize=(11, 5))
    ax = f.add_subplot(121)
    ax.bar(iter_list, hist_count, width=0.01)
    ax.set_xlabel("The count distribution", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax = f.add_subplot(122)
    ax.bar(iter_list, hist_error, width=0.01)
    ax.set_xlabel("The error distribution", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)

    # axs[1, 0].hist(y, bins=100, rwidth=0.8)
    # axs[1, 0].set_xlabel("y")
    # # axs[1,0].set_xlim([-0.8, 0.8])

    # axs[1, 1].hist(y_hat, bins=100, rwidth=0.8)
    # axs[1, 1].set_xlabel("y_hat")
    # axs[1,1].set_xlim([-0.8, 0.8])
    plt.tight_layout()
    writer.add_figure(f"{cv_index}_count_analysis", f)

    indexs = list(range(len(hist_count)))
    nonzero_index = [i for i in indexs if hist_count[i] > 0]
    hist_count = np.array(hist_count)[nonzero_index]
    hist_error = np.array(hist_error)[nonzero_index]
    # print(hist_count, hist_error)
    r, p = pearsonr(hist_count, hist_error)
    logging.info(
        f"Pearsonr(y,y_hat)={r:.3f}, p={p:.2e}",
    )


def cosine_similarity(a, b):
    res = (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
    return res.item()


def feature_statistics(df, cv_index, writer):
    y = np.array(df["y"])
    # img_fea = np.array(df["img_fea"])
    # num_fea = np.array(df["num_fea"])
    all_fea = np.array(df["last_fea"])
    hist_index, iter_list = get_hist(y, 0.0, 1.0, 0.02)

    hist_fea = {"Mean": [], "Variance": []}
    for key in hist_index.keys():
        if len(hist_index[key]) != 0:
            fea_in_bin = all_fea[hist_index[key], :]
            mean_fea = np.mean(fea_in_bin, axis=0)
            var_fea = np.var(fea_in_bin, axis=0)
            hist_fea["Mean"].append(mean_fea)
            hist_fea["Variance"].append(var_fea)
        else:
            hist_fea["Mean"].append(0)
            hist_fea["Variance"].append(0)

    hist_cosine = {"Mean": [], "Variance": []}
    bin0_index = 0

    for key in ["Mean", "Variance"]:
        fea0 = hist_fea[key][bin0_index]
        for i, crt_fea in enumerate(hist_fea[key]):
            if isinstance(crt_fea, int):
                hist_cosine[key].append(0)
            else:
                hist_cosine[key].append(
                    cosine_similarity(fea0.reshape(1, -1), crt_fea.reshape(1, -1))
                )

        f = plt.figure(dpi=200, figsize=(6, 5))
        ax = f.add_subplot(111)
        ax.bar(iter_list, hist_cosine[key], width=0.01)
        ax.set_ylabel(f"{key} cosine similarity")
        ax.set_xlabel("Target value")
        writer.add_figure(f"{cv_index}_{key}_feature_statistics", f)


def get_hist(y, min_value=0.0, maxs_value=1.0, step=0.01):
    sorted_id = sorted(range(len(y)), key=lambda k: y[k])
    y = np.array([y[i] for i in sorted_id])
    hist_index = {}
    iter_list = np.arange(min_value, maxs_value, step)
    for i in iter_list:
        start = i
        end = i + step
        hist_index[round(end, 2)] = np.where((y > start) & (y <= end))[0]
    return hist_index, iter_list


def pipline(log_dir="Logs/swint224_loss1"):
    logging_setting()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    writer = SummaryWriter(log_dir=log_dir)
    # read config
    for fold in glob.glob(f"{log_dir}/*/"):
        cv_index = os.path.basename(os.path.dirname(fold))
        args = parse_yaml(os.path.join(fold, "config.yaml"))
        setup_seed(args["M"]["seed"])
        data_list = parse_yaml(os.path.join(f"{log_dir}/{cv_index}.yaml"))
        test_list = data_list["test_list"]
        test_list = [os.path.join(args["D"]["data_dir"], i) for i in test_list]
        model, callback = get_model(args)
        loader = DataLoader(
            mpi_dataset(args, test_list),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            model.eval()
            epoch = args["M"]["crt_epoch"]
            model.load_state_dict(torch.load(args["M"]["best_weight_path"]))

            meters = {
                "y": Meter(),
                "yhat": Meter(),
                "name": Meter(),
                "lon": Meter(),
                "lat": Meter(),
                "ind": Meter(),
                "indhat": Meter(),
                "img_fea": Meter(),
                "num_fea": Meter(),
                "last_fea": Meter(),
            }
            for img, num, lbl, ind in loader:
                img = img.float().cuda()
                num = num.float().cuda()
                y = lbl["MPI3_fixed"].float().cuda()
                ind = ind.float().cuda()
                yhat, indhat = model(img, num)

                meters["y"].update(y)
                meters["yhat"].update(yhat)
                meters["name"].update(lbl["name"])
                meters["lon"].update(lbl["lon"])
                meters["lat"].update(lbl["lat"])
                meters["ind"].update(ind)
                meters["indhat"].update(indhat)
                meters["img_fea"].update(callback["img_fea"].data)
                meters["num_fea"].update(callback["num_fea"].data)
                meters["last_fea"].update(callback["last_fea"].data)

            r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
            rmse = math.sqrt(
                mean_squared_error(meters["yhat"].cat(), meters["y"].cat())
                .numpy()
                .item()
            )
            # acc = {"test/r2": r2, "test/rmse": rmse}

            logging.info(f"[test] Testing with {args['M']['best_weight_path']}")
            logging.info(f"[test] r2={r2:.3f} rmse={rmse:.4f}")

            # writer.add_scalar("Test/r2", acc["test/r2"], epoch)
            # writer.add_scalar("Test/rmse", acc["test/rmse"], epoch)
            # writer.add_histogram("Test/img_fea", callback["img_fea"].data, epoch)
            # writer.add_histogram("Test/num_fea", callback["num_fea"].data, epoch)

            df = {
                "name": meters["name"].cat(),
                "y": meters["y"].cat(),
                "yhat": meters["yhat"].cat(),
                "lon": meters["lon"].cat(),
                "lat": meters["lat"].cat(),
                "img_fea": meters["img_fea"].cat(),
                "num_fea": meters["num_fea"].cat(),
                "last_fea": meters["last_fea"].cat(),
            }

            # count_analysis(df, cv_index, writer)
            feature_statistics(df, cv_index, writer)

    writer.close()


if __name__ == "__main__":

    fire.Fire(pipline)
