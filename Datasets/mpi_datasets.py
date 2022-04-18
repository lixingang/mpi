import torch
import h5py
import numpy as np
import os,sys,glob
class mpi_dataset():
    def __init__(self, args, datalist):
        '''
        h5f: the h5 object
        '''
        self.datalist = datalist
        self.img_keys = args.img_keys
        self.num_keys = args.num_keys
        self.pos_keys = args.pos_keys
        self.label_keys = args.label_keys
 
    def __getitem__(self, i):
        data = torch.load(self.datalist[i])
        img = np.stack([data[k] for k in self.img_keys],-1)
        num = np.stack([data[k] for k in self.num_keys],-1)
        lbl = {k: np.squeeze(np.stack(data[k],-1)) for k in self.label_keys} 
        lbl["name"]=self.datalist[i]
        return (img.squeeze(),num.squeeze()), lbl

    def __len__(self):
        return len(self.datalist)


if __name__=='__main__': 
    pass
    # for y, fea_img, fea_num in train_loader:
    #     print(y.shape, fea_img.shape, fea_num.shape)







