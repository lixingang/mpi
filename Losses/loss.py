import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss
from Utils.base import *


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma**2
        return bmc_loss(pred, target, noise_var.cuda())


def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    pred = pred.unsqueeze(-1)
    target = target.unsqueeze(-1)
    logits = -(pred - target.T).pow(2) / (2 * noise_var)  # logit size: [batch, batch]

    loss = F.cross_entropy(
        logits, torch.arange(pred.shape[0]).cuda()
    )  # contrastive-like loss
    loss = (
        loss * (2 * noise_var).detach()
    )  # optional: restore the loss scale, 'detach' when noise is learnable

    return loss


from torch.distributions import MultivariateNormal as MVN


def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    I = torch.eye(pred.shape[-1])
    logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(
        target.unsqueeze(0)
    )  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))  # contrastive-like loss
    loss = (
        loss * (2 * noise_var).detach()
    )  # optional: restore the loss scale, 'detach' when noise is learnable

    return loss


def l1_l2_loss(pred, true, l1_weight, scores_dict):
    """
    Regularized MSE loss; l2 loss with l1 loss too.
    Parameters
    ----------
    pred: torch.floatTensor
        The model predictions
    true: torch.floatTensor
        The true values
    l1_weight: int
        The value by which to weight the l1 loss
    scores_dict: defaultdict(list)
        A dict to which scores can be appended.
    Returns
    ----------
    loss: the regularized mse loss
    """
    loss = F.mse_loss(pred, true)

    scores_dict["l2"].append(loss.item())

    if l1_weight > 0:
        l1 = F.l1_loss(pred, true)
        loss += l1
        scores_dict["l1"].append(l1.item())
    scores_dict["loss"].append(loss.item())

    return loss


class HEMLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.mse = nn.MSELoss()
        self.margin = margin

    def forward(self, pred, real):

        cond = torch.abs(real - pred) > self.margin

        if cond.long().sum() > 0:
            real = real[cond]
            pred = pred[cond]
            return self.mse(real, pred)
        else:
            return 0.0 * self.mse(real, pred)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=128, device=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        if self.device:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim).to(self.device)
            )
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )

        distmat.addmm_(mat1=x, mat2=self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.device:
            classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        labels = (labels).int()
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        return loss


# loss.py: Define the loss functions (here we only need the L1 loss)
import torch
import torch.nn.functional as F


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss.squeeze_()
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction="none")
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(
    inputs, targets, weights=None, activate="sigmoid", beta=0.2, gamma=1
):

    loss = (inputs - targets) ** 2
    loss *= (
        (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma
        if activate == "tanh"
        else (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    )
    if weights is not None:
        loss.squeeze_()
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(
    inputs, targets, weights=None, activate="sigmoid", beta=0.2, gamma=1
):
    loss = F.l1_loss(inputs, targets, reduction="none")
    loss *= (
        (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma
        if activate == "tanh"
        else (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    )
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.0):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss**2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.view(loss.size())
    loss = torch.mean(loss)
    return loss


if __name__ == "__main__":
    pass
    # torch.autograd.set_detect_anomaly(True)
    # bs = 30
    # real = torch.randint(1,13,(bs,)) + torch.normal(torch.zeros((bs,)),torch.ones((bs,)))
    # pred = real +  torch.normal(torch.zeros((bs,)),torch.ones((bs,)))
    # fea = torch.normal(torch.zeros((bs, 3)), torch.ones(bs, 3))
    # # print(fea.t().shape)
    # fea.requires_grad=True
    # loss_obj = marginloss()
    # loss = loss_obj(real, pred, fea)
    # loss.backward()
    # print(fea.grad)
