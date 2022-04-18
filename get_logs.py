import numpy as np
import os,argparse
import glob
import pandas as pd
import yaml
import re
class ParseYAML(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __init__(self, yaml_config):
        config = None
        with open(yaml_config, 'r') as f:
            config = yaml.load(f,Loader=yaml.Loader)
        for key in config:
            setattr(self, key, config[key])
            
    def __getattr__(self,name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
    def __setattr__(self,name,value):
        self[name]=value

def parse_log(path):
    f = open(path,'r')
    lines  = f.readlines()
    line = lines[-1].strip()
    pattern = re.compile(r'(?<=r2.)\d+\.?\d*')
    r2 = pattern.findall(line)[0]
    pattern = re.compile(r'(?<=rmse.)\d+\.?\d*')
    rmse = pattern.findall(line)[0]
    pattern = re.compile(r'(?<=mape.)\d+\.?\d*')
    mape = pattern.findall(line)[0]
    f.close()
    return float(r2),float(rmse), float(mape)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(f'--log_dir', default="mpi",type=str)
args = parser.parse_args()

def compute_average(res_list):
    res = [float(i) for i in res_list]
    res = np.array(res)
    return np.average(res)

if __name__=='__main__':
    assert args.log_dir!=None
    items = {"name":[],"seed":[],"r2":[],"rmse":[],"mape":[]}
    log_list = glob.glob(f"Logs/{args.log_dir}/*")
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
    items["r2"].append(compute_average(items["r2"]))
    items["rmse"].append(compute_average(items["rmse"]))
    items["mape"].append(compute_average(items["mape"]))
    #print(items)
    df = pd.DataFrame(items,index=None)
    df.to_csv(f"Logs/{args.log_dir}/STAT_{args.log_dir}.csv",index=False)

    


