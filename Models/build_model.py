# from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
from .swin_transformer import SwinTransformer
from Models.fds import FDS
from Losses.loss import *


class SwinTransformerForMpi(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        self.channel_enhance = nn.Sequential(
            nn.Conv2d(self.in_chans, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, self.in_chans, 1, 1, 0),
        )

        self.CA_layer = SELayer(self.in_chans, 4)

    def forward(self, x, mask=None):

        x = self.channel_enhance(x)
        x = self.CA_layer(x)
        x = self.patch_embed(x)

        # assert mask is not None
        # B, L, _ = x.shape

        # mask_tokens = self.mask_token.expand(B, L, -1)
        # w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        # x = x * (1.0 - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # B, C, L = x.shape
        # H = W = int(L**0.5)
        # x = x.reshape(B, C, H, W)
        return x


class NumLayers(nn.Module):
    def __init__(self, in_chans, out_chans=48):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_layers = nn.Sequential(
            nn.Linear(self.in_chans, self.out_chans),
            nn.BatchNorm1d(self.out_chans),
            nn.Linear(self.out_chans, self.out_chans),
            nn.BatchNorm1d(self.out_chans),
            nn.Linear(self.out_chans, self.out_chans),
        )

    def forward(self, x):
        x = self.num_layers(x)
        return x


class MpiForecastModel(nn.Module):
    def __init__(self, img_encoder, num_encoder=NumLayers(11, 48)):
        super().__init__()
        self.img_encoder = img_encoder
        self.num_encoder = num_encoder
        num_features = img_encoder.num_features

        self.neck = nn.Sequential(
            nn.Linear(num_features + num_encoder.out_chans, num_features),
            nn.Linear(num_features, num_features),
        )

        self.head = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, 1),
        )
        """
        Self Supervised (SSP)
        """
        self.head_num = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, 11),
        )
        # self.head_img = nn.Sequential(
        #     # nn.Linear(num_features, num_features),
        #     nn.BatchNorm1d(num_features),
        #     nn.Linear(
        #         num_features,
        #         int(img_size / 4 * img_size / 4 * self.mask_ratio)
        #         * self.img_encoder.embed_dim,
        #     ),
        # )

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.encoder.num_features,
        #         out_channels=self.encoder_stride**2 * 3,
        #         kernel_size=1,
        #     ),
        #     nn.PixelShuffle(self.encoder_stride),
        # )

        # self.in_chans = self.encoder.in_chans
        # self.patch_size = self.encoder.patch_size
        self.apply(self._init_weights)
        """
        FDS
        """
        self.fds_config1 = dict(
            feature_dim=self.img_encoder.num_features,
            start_update=0,
            start_smooth=10,
            kernel="gaussian",
            ks=10,
            sigma=2,
        )
        self.fds_config2 = dict(
            feature_dim=self.num_encoder.out_chans,
            start_update=0,
            start_smooth=10,
            kernel="gaussian",
            ks=5,
            sigma=2,
        )

        self.FDS1 = FDS(**self.fds_config1)
        self.FDS2 = FDS(**self.fds_config2)

    def forward(self, img, num, aux={}):
        f_img = self.img_encoder(img)
        if len(aux) != 0:
            if aux["epoch"] >= self.fds_config1["start_smooth"]:
                f_img = self.FDS1.smooth(f_img, aux["label"], aux["epoch"])
        f_num = self.num_encoder(num)
        if len(aux) != 0:
            if aux["epoch"] >= self.fds_config2["start_smooth"]:
                f_num = self.FDS2.smooth(f_num, aux["label"], aux["epoch"])

        fea = torch.cat((f_img, f_num), dim=1)
        fea = self.neck(fea)
        x_hat = self.head(fea)
        num_rec = self.head_num(fea)

        # img_rec = self.head_img(fea)

        # losses = {}
        # losses["num_rec"] = weighted_huber_loss(num, num_rec)
        # loss_img_rec = (
        #     F.l1_loss(img, num_rec, reduction="none") / self.num_encoder.in_chans
        # )

        return num_rec, x_hat

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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
