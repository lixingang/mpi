import torch
import numpy as np
import random
import time
import yaml
import re
from collections import namedtuple
import munch


def asleast1d(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    assert len(arr.shape) <= 2, "arr.shape>2"
    return arr.reshape(len(max(arr.shape)), 1)


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
    torch.backends.cudnn.benchmark = False


def randnorepeat(m, n):
    p = list(range(n))
    d = random.sample(p, m)
    return d


# def split_train_test(data_list, ratio=[0.6, 0.2, 0.2]):
#     idx = list(range(len(data_list)))
#     assert len(ratio) >= 2 and len(ratio) <= 3, "请确认ratio>=2 and <=3"
#     assert np.sum(np.array(ratio)) == 1.0, "请确认ratio总和为1"
#     random.shuffle(idx)
#     slice1 = int(len(idx) * ratio[0])
#     slice2 = int(len(idx) * (ratio[1] + ratio[0]))
#     if len(ratio) == 2:
#         return data_list[:slice1], data_list[slice1:slice2]
#     else:
#         return data_list[:slice1], data_list[slice1:slice2], data_list[slice2:]


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


# def dict2namedtuple(obj):
#     if isinstance(obj, dict):
#         for key, value in obj.items():
#             obj[key] = dict2namedtuple(value)
#         return namedtuple("GenericDict", obj.keys())(**obj)
#     elif isinstance(obj, list):
#         return [dict2namedtuple(item) for item in obj]
#     else:
#         return obj


def parse_yaml(path):
    # config = None
    # with open(path, "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = bunch.Bunch().fromDict(config)
    with open(path, "r") as f:
        config = munch.Munch.fromYAML(f, Loader=yaml.FullLoader)
    return config


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)
    return 1


"""
ssp
"""


def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def patchify(self, imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = self.patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x


def unpatchify(self, x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = self.patch_embed.patch_size[0]
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


if __name__ == "__main__":
    args = parse_yaml("/home/lxg/mpi/swint.yaml")
    # print(args)
    print(args.M.model)
    save_yaml(args, "saveconfig.yaml")
