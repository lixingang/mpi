# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
import os,sys
if __name__=="__main__":
    sys.path.append("..")
    from mobilenetv3 import MobileNetV3_Small
    from Utils.clock import clock
    from config import train_config
else:
    from Models.mobilenetv3 import MobileNetV3_Small
    from Utils.clock import clock
    from config import train_config
    
class Net(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.Cnet = MobileNetV3_Small(in_channel=len(args.img_keys) , out_channel=128)
        self.Lnet = nn.Sequential(
            nn.Linear(len(args.num_keys), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.fclayer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.Softmax(dim=1)
        )
        
    def forward(self,img, num):
        # print(img.size())
        img = img.permute(0,3,1,2)
        img = self.Cnet(img)
        num = self.Lnet(num.float())
        x = torch.cat((img,num),1)
        x = self.fclayer(x)
        # print(x)
        
        return torch.squeeze(x)

        

if __name__=="__main__":

    A=torch.rand(60,10,255,255)
    B=torch.rand(60,19)  
    Label=torch.rand(60,1)
    model = Net()
    res = model((A,B))
    print(res.shape)
    print(res)
    
    