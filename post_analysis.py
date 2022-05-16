import numpy as np
import glob
import os
import pandas as pd
import argparse
from Utils.base import parse_yaml, parse_log


def get_logs(logname):
    items = {"name": [], "seed": [], "r2": [], "rmse": [], "mape": []}
    log_list = glob.glob(f"Logs/{logname}/*")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(f"--name", default="", type=str)
    args = parser.parse_args()
    get_logs(args.name)
