import torch
import numpy as np
import random
import time
import yaml
import re


"""
HOOK: 
SaveOutput
"""


class SaveOutput:
    def __init__(self):
        self.data = None

    def __call__(self, module, module_in, module_out):
        # print(module_out.clone().detach().shape)
        self.data = module_out.clone().detach().cpu()

        return None


"""
小模块:
setup_seed
split_train_test
Meter
"""


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def randnorepeat(m, n):
    p = list(range(n))
    d = random.sample(p, m)
    return d


def split_train_test(data_list, ratio=[0.6, 0.2, 0.2]):
    idx = list(range(len(data_list)))
    assert len(ratio) >= 2 and len(ratio) <= 3, "请确认ratio>=2 and <=3"
    assert np.sum(np.array(ratio)) == 1.0, "请确认ratio总和为1"
    random.shuffle(idx)
    slice1 = int(len(idx) * ratio[0])
    slice2 = int(len(idx) * (ratio[1] + ratio[0]))
    if len(ratio) == 2:
        return data_list[:slice1], data_list[slice1:slice2]
    else:
        return data_list[:slice1], data_list[slice1:slice2], data_list[slice2:]


def split_train_valid(data_list, ratio=[0.6, 0.4]):
    idx = list(range(len(data_list)))
    assert len(ratio) == 2, "请确认ratio长度为2，该函数仅用于划分训练、验证集"
    assert np.sum(np.array(ratio)) == 1.0, "请确认ratio总和为1"
    random.shuffle(idx)
    slice1 = int(len(idx) * ratio[0])
    slice2 = int(len(idx) * (ratio[1] + ratio[0]))
    return data_list[:slice1], data_list[slice1:slice2]


class Meter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.values = []
        self.reset()

    def reset(self):
        self.values = []

    def update(self, val):
        if isinstance(val, list):
            pass
        elif isinstance(val, np.ndarray):
            val = val.tolist()
        elif torch.is_tensor(val):
            if val.is_cuda:
                val = val.detach().cpu()
            # val = val.tolist()

        self.values.append(val)

    def avg(self):
        try:
            values_flatten = np.asarray(self.values).flatten()
            return np.average(values_flatten)
        except:
            print("An exception occurred, checkout the type of Meter")

    def cat(self, dim=0):
        assert len(self.values) != 0, "Meter内的元素个数为0"
        if isinstance(self.values[0], list):
            return list(np.concatenate(self.values).flatten())
        else:
            return torch.cat(self.values, dim=dim).squeeze()


"""
解析数据
parse_yaml
parse_log
"""


def parse_yaml(path):
    config = None
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def parse_log(path):

    f = open(path, "r")
    lines = f.readlines()
    line = lines[-1].strip()
    pattern = re.compile(r"(?<=r2.)\d+\.?\d*")
    r2 = pattern.findall(line)[0]
    pattern = re.compile(r"(?<=mse.)\d+\.?\d*")
    rmse = pattern.findall(line)[0]
    # pattern = re.compile(r'(?<=mape.)\d+\.?\d*')
    # mape = pattern.findall(line)[0]
    mape = 0
    f.close()
    return float(r2), float(rmse), float(mape)
