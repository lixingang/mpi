# from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
from .swin_transformer import SwinTransformer
from Models.fds import FDS
from Losses.loss import *


class MpiForecastModel(SwinTransformer):
    def __init__(self, **kargs):
        super().__init__(**kargs)

        self.num_layers = NumLayers(in_chans=self.in_chans[1])

        self.neck = nn.Sequential(
            nn.Linear(self.num_features + 48, self.num_features),
            nn.Linear(self.num_features, self.num_features),
        )

        # head
        self.head = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, 1),
        )
        self.head_num = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_features, 11),
        )

        self.apply(self._init_weights)

        self.fds_config1 = dict(
            feature_dim=self.num_features,
            start_update=0,
            start_smooth=10,
            kernel="gaussian",
            ks=10,
            sigma=2,
        )
        self.fds_config2 = dict(
            feature_dim=48,
            start_update=0,
            start_smooth=10,
            kernel="gaussian",
            ks=5,
            sigma=2,
        )

        self.FDS1 = FDS(**self.fds_config1)
        self.FDS2 = FDS(**self.fds_config2)
        # self.indicator_weights = torch.nn.Parameter(
        #     torch.tensor(
        #         [1/6.0,1/6.0,1/6.0,1/6.0,1/18.0,1/18.0,1/18.0,1/18.0,1/18.0,1/18.0,
        #         ]
        #     )
        # )

        # self.calibrated_layer = nn.Sequential(
        #     nn.Linear(self.num_features, self.num_features),
        #     nn.ReLU(),
        #     nn.Linear(self.num_features, self.num_features),
        # )

    def forward(self, img, num, aux={}):
        x = self.channel_enhance(img)
        x = self.CA_layer(x)
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        f_img = torch.flatten(x, 1)
        f_num = self.num_layers(num)
        if len(aux) != 0:
            if aux["epoch"] >= self.fds_config1["start_smooth"]:
                f_img = self.FDS1.smooth(f_img, aux["label"], aux["epoch"])

        if len(aux) != 0:
            if aux["epoch"] >= self.fds_config2["start_smooth"]:
                f_num = self.FDS2.smooth(f_num, aux["label"], aux["epoch"])
        fea = torch.cat((f_img, f_num), dim=1)
        fea = self.neck(fea)
        x_hat = self.head(fea)
        num_hat = self.head_num(fea)
        # xi_hat = self.head2(fea)
        loss_num_rec = torch.mean(weighted_huber_loss(num_hat, num))

        # self.indicator_weights = torch.nn.Parameter(self.indicator_weights)
        # x = torch.sum(torch.mul(self.indicator_weights, ind_hat), dim=-1)
        return x_hat, loss_num_rec, fea


class NumLayers(nn.Module):
    def __init__(self, in_chans, out_chans=48):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Linear(self.in_chans, self.out_chans),
            nn.BatchNorm1d(self.out_chans),
            nn.Linear(self.out_chans, self.out_chans),
            nn.BatchNorm1d(self.out_chans),
            nn.Linear(self.out_chans, self.out_chans),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MaskGenerator:
    def __init__(
        self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask
