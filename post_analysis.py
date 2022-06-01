import numpy as np
import glob
import os
import pandas as pd
import fire
import matplotlib.pyplot as plt
from Utils.base import parse_yaml, parse_log


def get_logs(logname):
    items = {"name": [], "seed": [], "r2": [], "rmse": [], "mape": []}
    log_list = glob.glob(f"Logs/{logname}/*/")
    for log in log_list:
        cfg = parse_yaml(os.path.join(log, "config.yaml"))
        r2, rmse, mape = parse_log(os.path.join(log, "run.log"))
        items["name"].append(log)
        items["seed"].append(cfg["M"]["seed"])
        items["r2"].append(r2)
        items["rmse"].append(rmse)
        items["mape"].append(mape)
    items["name"].append("Average")
    items["seed"].append("-1")
    items["r2"].append(np.average([float(i) for i in items["r2"]]))
    items["rmse"].append(np.average([float(i) for i in items["rmse"]]))
    items["mape"].append(np.average([float(i) for i in items["mape"]]))
    # print(items)
    df = pd.DataFrame(items, index=None)
    df.to_csv(f"Logs/{logname}/STAT_{logname}.csv", index=False)


def get_indicator(logname="swint_config_base", mode="separate"):
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
        plt.figure(figsize=(12, 20), dpi=300)
        for i, log in enumerate(log_list):
            print(f"正在输出{log}日志中的indicator分布...")
            ind_lbl = pd.read_csv(os.path.join(log, "weight_indicator.csv")).to_numpy()
            ind_hat = pd.read_csv(os.path.join(log, "weight_features.csv")).to_numpy()
            ind_diff = np.abs(ind_hat - ind_lbl)
            plt.subplot(5, 1, i + 1)
            plt.boxplot(ind_diff)
            plt.xticks(range(1, 11), columns)
            plt.title(log)
            plt.tight_layout()
        plt.savefig("indicator_diff_abs.jpg")


if __name__ == "__main__":
    fire.Fire()
