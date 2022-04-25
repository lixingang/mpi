# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
import os,sys

from Utils.clock import clock
from Models.mobilenetv3 import MobileNetV3_Small
from Utils.clock import clock
    
class Net(nn.Module):
    def __init__(self, args):

        super().__init__()
        # self.Cnet = MobileNetV3_Small(in_channel=len(args.img_keys) , out_channel=128)
        self.Lnet1 = nn.Sequential(
            nn.Linear(len(args.img_keys)*20, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.Lnet2 = nn.Sequential(
            nn.Linear(len(args.num_keys), 512),
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
            nn.Linear(1024, 1),
            # nn.Softmax(dim=1)
        )
        
    def forward(self,img, num):
        # print(img.size())
        img = img.view(img.shape[0],-1)
        img = self.Lnet1(img.float())
        num = self.Lnet2(num.float())
        fea = torch.cat((img,num),1)
        # fea = img
        x = self.fclayer(fea)
        # print(x)
        
        return torch.squeeze(x), fea

        

if __name__=="__main__":

    A=torch.rand(60,10,25)
    B=torch.rand(60,19)  
    Label=torch.rand(60,1)
    model = Net()
    res = model((A,B))
    print(res.shape)
    print(res)
    
    