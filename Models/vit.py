import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from Models.fds import FDS

# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# helpers

fds_config = dict(
    feature_dim=128, start_update=0, start_smooth=1, kernel="gaussian", ks=5, sigma=2
)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        num_patches,
        patch_dim,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        # X = [args.batch_size, num_patches, patch_dim]
        self.patch_dim = patch_dim
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_classes), nn.Sigmoid()
        )

        self.FDS = FDS(**fds_config)

    def forward(self, img, num, aux={}):
        # 把num扩充到20维
        # 拼接img and x.shape=[args.batch_size, num_patches, patch_dim]
        # x
        num = repeat(num, "b c -> b d c", d=self.patch_dim)
        data = torch.cat((img, num), dim=-1)
        data = rearrange(data, "b d p -> b p d").float()

        x = self.to_patch_embedding(data)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        # print(x.shape)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)

        if len(aux) != 0:
            if aux["epoch"] >= fds_config["start_smooth"]:
                x = self.FDS.smooth(x, aux["label"], aux["epoch"])

        ind_hat = self.mlp_head(x)

        ind_weights = torch.tensor(
            [
                1 / 6.0,
                1 / 6.0,
                1 / 6.0,
                1 / 6.0,
                1 / 18.0,
                1 / 18.0,
                1 / 18.0,
                1 / 18.0,
                1 / 18.0,
                1 / 18.0,
            ]
        ).cuda()

        x = torch.sum(torch.mul(ind_weights, ind_hat), dim=-1)

        return x, ind_hat


if __name__ == "__main__":
    v = ViT(
        num_patches=34,
        patch_dim=20,
        num_classes=1,
        dim=1024,
        depth=3,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

    img = torch.randn(30, 34, 20)

    preds = v(img)  # (1, 1000)
    print(preds.shape)
"""
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)

"""
