import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F


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

    def forward(self, pred,real):

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
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        

        distmat.addmm_( mat1=x, mat2=self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.device: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        labels = (labels).int()
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


# class marginloss(nn.Module):
#     def __init__(self, ):
#         super().__init__()
#         self.ms = np.load("/root/data/model/dataset/ms0103/ms_y.npy",allow_pickle=True)
#         # print(ms)
        
#     def forward(self, pred, real,fea):
#         if len(fea)==1:
#             return 0
#         pred = pred*self.ms[1]+self.ms[0] #恢复到归一化之前的y值
#         real = real*self.ms[1]+self.ms[0]
#         class_value = torch.arange(0,5) #根据真值值域分成n类
#         real_int = torch.floor(real/3) #取floor值
#         distance = 0 #特征距离
#         for i in class_value:
#             mask = torch.where(torch.eq(real_int, i), 1, 0)
#             if torch.count_nonzero(mask)==0:
#                 distance+=0
#             else:
#                 X = fea[torch.squeeze(mask>0)] 
#                 G = torch.mm(X,X.t())
#                 G_diag = torch.diag(G)
#                 H =  G_diag.repeat( len(X),1)
#                 HG = H+H.t()-2*G+1e-8
#                 # print(i,HG)
#                 distance += ( torch.sum(torch.sqrt(HG)) / len(X) / len(real))
#             # print(distance)
#             # print(distance)
#         return distance


'''
def compute_squared_EDM_method4(X):
  # 获得矩阵都行和列，因为是行向量，因此一共有n个向量
  n,m = X.shape
  # 计算Gram 矩阵
  G = np.dot(X,X.T)
  # 因为是行向量，n是向量个数,沿y轴复制n倍，x轴复制一倍
  H = np.tile(np.diag(G), (n,1))
  return np.sqrt(H + H.T - 2*G)
'''


if __name__=="__main__":
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

