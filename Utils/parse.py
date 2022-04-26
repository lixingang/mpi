import yaml
import numpy as np
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
    pattern = re.compile(r'(?<=mse.)\d+\.?\d*')
    rmse = pattern.findall(line)[0]
    # pattern = re.compile(r'(?<=mape.)\d+\.?\d*')
    # mape = pattern.findall(line)[0]
    mape=0
    f.close()
    return float(r2),float(rmse), float(mape)
