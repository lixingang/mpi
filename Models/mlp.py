# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
import os
import sys
sys.path.append("/home/lxg/data/mpi/")
from Models.mobilenetv3 import MobileNetV3_Small
from Models.fds import FDS

fds_config = dict(feature_dim=1024, start_update=0,
                  start_smooth=1, kernel='gaussian', ks=5, sigma=2)


class MLP(nn.Module):
    def __init__(self, img_in_channel, num_in_channel):

        super().__init__()

        self.FDS = FDS(**fds_config)
        # self.Cnet = MobileNetV3_Small(in_channel=len(args.img_keys) , out_channel=128)
        self.Lnet1 = nn.Sequential(
            nn.Linear(img_in_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.Lnet2 = nn.Sequential(
            nn.Linear(num_in_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.fclayer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10),
            nn.ReLU(inplace=True),
            # nn.Softmax(dim=1)
        )

    def forward(self, img, num, aux={}):
        img = img.view(img.shape[0], -1)
        img = self.Lnet1(img.float())
        num = self.Lnet2(num.float())
        fea = torch.cat((img, num), 1)

        if len(aux) != 0:
            if aux["epoch"] >= fds_config['start_smooth']:
                fea = self.FDS.smooth(fea, aux["label"], aux["epoch"])

        x = self.fclayer(fea)
        indicator_weights = torch.tensor([
            1/6.0, 1/6.0, 1/6.0, 1/6.0,
            1/18.0, 1/18.0, 1/18.0, 1/18.0, 1/18.0, 1/18.0,
        ]).cuda()
        x = torch.sum(torch.mul(indicator_weights, x), dim=-1)

        return x


'''
MLP(
  (FDS): FDS()
  (Lnet1): Sequential(
    (0): Linear(in_features=500, out_features=512, bias=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=512, out_features=512, bias=True)
    (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (Lnet2): Sequential(
    (0): Linear(in_features=19, out_features=512, bias=True)
    (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=512, out_features=512, bias=True)
    (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (fclayer): Sequential(
    (0): Linear(in_features=1024, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Linear(in_features=1024, out_features=1, bias=True)
  )
)

'''
if __name__ == "__main__":

    A = torch.rand(60, 20, 25)
    B = torch.rand(60, 19)


    model = MLP(20*25,19)
    res = model(A, B)
    print(res.shape)
