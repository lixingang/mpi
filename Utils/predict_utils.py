import matplotlib.pyplot as plt
import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import fire
from torchmetrics.functional import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from torch.utils.tensorboard import SummaryWriter

# import in-project packages
from Models import *
from Datasets.mpi_datasets import mpi_dataset
from main import prepare_model
from Utils.base import parse_yaml, setup_seed


def get_hist(y, min_value=0.0, maxs_value=0.9, step=0.01):
    sorted_id = sorted(range(len(y)), key=lambda k: y[k])
    y = np.array([y[i] for i in sorted_id])
    hist_index = {}
    iter_list = np.arange(min_value, maxs_value, step)
    for i in iter_list:
        start = i
        end = i + step
        hist_index[round(end, 2)] = np.where((y > start) & (y <= end))[0]
    return hist_index, iter_list


def chart_1_fold_result(df, cv_index, writer):
    y = df["y"]
    yhat = df["yhat"]
    f = plt.figure(dpi=150, figsize=(6, 5))
    ax = f.add_subplot(1, 1, 1)
    ax.scatter(y, yhat)
    ax.set_xlabel("y")
    ax.set_ylabel("yhat")
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0, 0.8])
    writer.add_figure(f"{cv_index}/chart_1_fold_result", f)


def count_analysis(df, cv_index, writer):
    y = np.array(df["y"])
    y_hat = np.array(df["yhat"])
    diff = np.abs(y - y_hat)
    hist_index, iter_list = get_hist(y, 0.0, 0.9, 0.05)
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
    writer.add_figure(f"{cv_index}/count_analysis", f)

    indexs = list(range(len(hist_count)))
    nonzero_index = [i for i in indexs if hist_count[i] > 0]
    hist_count = np.array(hist_count)[nonzero_index]
    hist_error = np.array(hist_error)[nonzero_index]
    # print(hist_count, hist_error)
    r, p = pearsonr(hist_count, hist_error)
    print(
        f"Pearsonr(y,y_hat)={r:.3f}, p={p:.2e}",
    )


def cosine_similarity(a, b):
    res = (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
    return res.item()


def feature_statistics(df, cv_index, writer):
    y = np.array(df["y"])
    # img_fea = np.array(df["img_fea"])
    # num_fea = np.array(df["num_fea"])
    all_fea = np.array(df["gp_feat"])
    hist_index, iter_list = get_hist(y, 0.0, 0.9, 0.02)

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
        writer.add_figure(f"{cv_index}/{key}_feature_statistics", f)


def plot_ind(logname="swint_config_base", mode="separate"):
    columns = [
        "Child mortality",
        "Nutrition",
        "School attendance",
        "Years of schooling",
        "Electricity",
        "Drinking water",
        "Sanitation",
        "Housing",
        "Cooking fuel",
        "Assets",
    ]
    assert mode in ["separate", "total"]
    if mode == "separate":
        log_list = glob.glob(f"Logs/{logname}/*/")
        plt.figure(figsize=(10, 25), dpi=300)

        for i, log in enumerate(log_list):
            print(f"正在输出{log}日志中的indicator分布...")
            ind_lbl = pd.read_csv(os.path.join(log, "ind.csv")).to_numpy()
            ind_hat = pd.read_csv(os.path.join(log, "ind_hat.csv")).to_numpy()

            ind_diff = np.abs(ind_hat - ind_lbl)
            plt.subplot(5, 1, i + 1)
            plt.boxplot(ind_diff)
            plt.xticks(range(1, 11), columns)
            plt.title(log)
            plt.tight_layout()

        plt.savefig(os.path.join(log, "indicator_diff_abs.jpg"))

        plt.figure(figsize=(25, 10), dpi=300)
        for i, log in enumerate(log_list):
            print(f"正在输出{log}日志中的决定系数r2...")
            ind_lbl = pd.read_csv(os.path.join(log, "ind.csv"))
            ind_hat = pd.read_csv(os.path.join(log, "ind_hat.csv"))

            ind_diff = np.abs(ind_hat - ind_lbl)
            for i, c in enumerate(columns):
                plt.subplot(2, 5, i + 1)
                x, y = ind_lbl[c], ind_hat[c]
                # linear_model = LinearRegression()
                # linear_model.fit(x, y)
                # y_p = linear_model.predict(x)

                r2 = r2_score(x, y)
                plt.scatter(x, y)
                plt.xlabel("True")
                plt.ylabel("Predict")
                plt.title(c, fontsize=16)
                # plt.plot(x, y_p, color="r")
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.text(0.1, 0.9, f"$r^2$={round(r2,3)}", fontsize=16)
                plt.tight_layout()

            plt.savefig(os.path.join(log, "indicator_r2.jpg"))
            break


if __name__ == "__main__":

    fire.Fire(pipline)
