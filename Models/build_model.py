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
        self.mask_ratio = 0.05
        # num
        self.num_layers = NumLayers(in_chans=self.in_chans[1])
        # img
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1, int(self.img_size**2 / (self.patch_size**2)), self.embed_dim
            ),
            requires_grad=False,
        )  # fixed sin-cos embedding
        # self.decoder_pos_embed = nn.Parameter(
        #     torch.zeros(
        #         1,
        #         int((self.img_size**2 / (self.patch_size**2)) * self.mask_ratio),
        #         self.num_features,
        #     ),
        #     requires_grad=False,
        # )  # fixed sin-cos embedding
        # concat
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
        self.head_img = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            # nn.BatchNorm1d(self.num_features),
            nn.Linear(
                self.num_features,
                int(
                    self.img_size
                    * self.img_size
                    / self.patch_size
                    / self.patch_size
                    * self.mask_ratio
                )
                * self.embed_dim,
            ),
        )

        self.initialize_weights()
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
        # 属性特征提取
        f_num = self.num_layers(num)
        # 图像特征提取
        x = self.channel_enhance(img)
        x = self.CA_layer(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed

        if self.ape:
            x = x + self.absolute_pos_embed
        # x = self.pos_drop(x)
        x, img_rec_target, id_restore = random_masking(x, self.mask_ratio)
        self.de_pos_embed = torch.gather(
            self.pos_embed.repeat(x.size(0), 1, 1),
            dim=1,
            index=id_restore.unsqueeze(-1).repeat(1, 1, x.size(-1)),
        )

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        f_img = torch.flatten(x, 1)

        # 特征平滑
        if len(aux) != 0:
            if aux["epoch"] >= self.fds_config1["start_smooth"]:
                f_img = self.FDS1.smooth(f_img, aux["label"], aux["epoch"])

        if len(aux) != 0:
            if aux["epoch"] >= self.fds_config2["start_smooth"]:
                f_num = self.FDS2.smooth(f_num, aux["label"], aux["epoch"])

        # 图像、属性特征融合
        fea = torch.cat((f_img, f_num), dim=1)
        fea = self.neck(fea)

        # 预测和重建
        x_hat = self.head(fea)
        num_hat = self.head_num(fea)
        img_hat = self.head_img(fea)
        img_hat = img_hat.view(img_hat.shape[0], -1, self.embed_dim)
        img_hat = img_hat + self.de_pos_embed
        # xi_hat = self.head2(fea)

        loss_img_rec = torch.mean(weighted_huber_loss(img_hat, img_rec_target))
        loss_num_rec = torch.mean(weighted_huber_loss(num_hat, num))

        # self.indicator_weights = torch.nn.Parameter(self.indicator_weights)
        # x = torch.sum(torch.mul(self.indicator_weights, ind_hat), dim=-1)
        return x_hat, (loss_num_rec, loss_img_rec), fea

    def initialize_weights(self):

        # self.decoder_pos_embed = get_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1],
        #     ,
        # )

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int((self.img_size**2 / (self.patch_size**2)) ** 0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1],
        #     int((self.img_size**2 / (self.patch_size**2)) ** 0.5),
        # )
        # self.decoder_pos_embed.data.copy_(
        #     torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        # )
        self.apply(self._init_weights)


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


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_extract = int(x.shape[1] * mask_ratio)

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_extract = ids_shuffle[:, :len_extract]

    rec_target = torch.gather(x, dim=1, index=ids_extract.unsqueeze(-1).repeat(1, 1, D))
    x_masked = torch.scatter(
        x,
        dim=1,
        index=ids_extract.unsqueeze(-1).repeat(1, 1, D),
        src=torch.zeros(ids_extract.unsqueeze(-1).repeat(1, 1, D).shape).cuda(),
    )
    # generate the binary mask: 0 is keep, 1 is remove
    # mask = torch.ones([N, L], device=x.device)
    # mask[:, :len_keep] = 0
    # # unshuffle to get the binary mask
    # mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, rec_target, ids_extract


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
