import numpy as np
import glob,os
import pandas as pd
import argparse
from Utils.parse import ParseYAML,parse_log
def get_logs(model_name):
    items = {"name":[],"seed":[],"r2":[],"rmse":[],"mape":[]}
    log_list = glob.glob(f"Logs/{model_name}/*")
    for log in log_list:
        cfg = ParseYAML(os.path.join(log,"config.yaml"))
        r2,rmse,mape = parse_log(os.path.join(log,"run.log"))
        items["name"].append(log)
        items["seed"].append(cfg.seed)
        items["r2"].append(r2)
        items["rmse"].append(rmse)
        items["mape"].append(mape)
    items["name"].append("Average")
    items["seed"].append("-1")
    items["r2"].append(np.average([float(i) for i in items["r2"]]))
    items["rmse"].append(np.average([float(i) for i in items["rmse"]]))
    items["mape"].append(np.average([float(i) for i in items["mape"]]))
    #print(items)
    df = pd.DataFrame(items,index=None)
    df.to_csv(f"Logs/{model_name}/STAT_{model_name}.csv",index=False)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(f'--model_name', default="",type=str)
    args = parser.parse_args()
    get_logs(args.model_name)