import torch
import h5py
import numpy as np
import os
import sys
import glob
import pandas as pd

sys.path.append("/home/lxg/data/mpi/")
from Utils.base import parse_yaml


class mpi_dataset:
    def __init__(self, args, datalist):
        """
        h5f: the h5 object
        """
        self.datalist = datalist
        self.img_keys = args["D"]["img_keys"]
        self.num_keys = args["D"]["num_keys"]
        self.label_keys = args["D"]["label_keys"]
        self.indicator_keys = args["D"]["indicator_keys"]

    def __getitem__(self, i):
        data = torch.load(self.datalist[i])

        img = np.stack([data[k] for k in self.img_keys], -1)
        img = np.transpose(img, (2, 0, 1))
        num = np.stack([data[k] for k in self.num_keys], -1)
        lbl = {k: np.squeeze(np.stack(data[k], -1)) for k in self.label_keys}
        ind = {k: np.squeeze(np.stack(data[k], -1)) for k in self.indicator_keys}
        ind = np.stack([ind[k] for k in ind.keys()], -1)
        lbl["name"] = self.datalist[i]

        return img.squeeze(), num.squeeze(), lbl, ind

    def __len__(self):
        return len(self.datalist)


if __name__ == "__main__":
    args = parse_yaml("config.yaml")
    data_list = np.array(glob.glob(f"Data/input_data/*"))
    ds = mpi_dataset(args, data_list)
    loader = torch.utils.data.DataLoader(
        mpi_dataset(args, data_list),
        batch_size=args["M"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    for img, num, lbl, ind in loader:
        print(img.shape)
        print(num.shape)
        print(ind.shape)
        break
        # print(ind)
