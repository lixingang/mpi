import yaml
import numpy as np
import re


def parse_yaml(path):
    config = None
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def parse_log(path):

    f = open(path, 'r')
    lines = f.readlines()
    line = lines[-1].strip()
    pattern = re.compile(r'(?<=r2.)\d+\.?\d*')
    r2 = pattern.findall(line)[0]
    pattern = re.compile(r'(?<=mse.)\d+\.?\d*')
    rmse = pattern.findall(line)[0]
    # pattern = re.compile(r'(?<=mape.)\d+\.?\d*')
    # mape = pattern.findall(line)[0]
    mape = 0
    f.close()
    return float(r2), float(rmse), float(mape)


if __name__ == '__main__':
    import argparse
    from collections import namedtuple
    p = parse_yaml("/home/lxg/data/mpi/config.yaml")

    p['MAIN']['model'] = 'vv3'
    # print(args.main)

    with open('config1.yaml', 'w') as f:
        yaml.dump(p, f)
