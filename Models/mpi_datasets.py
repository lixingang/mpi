if __name__=='__main__':
    sys.path.append("..")
import torch
import h5py
import numpy as np
import os,sys,glob
from Utils.clock import clock
class mpi_dataset():
    def __init__(self, args, h5path):
        '''
        h5f: the h5 object
        '''
        self.h5f = h5py.File(h5path,"r")
        self.img_keys = args.img_keys
        self.num_keys = args.num_keys
        self.pos_keys = args.pos_keys
        self.label_keys = args.label_keys
 
    def __getitem__(self, i):
        img = self._get_item_from_keys(self.img_keys, i)
        num = self._get_item_from_keys(self.num_keys, i)
        label = self._get_item_from_keys(self.label_keys, i)
        return label.squeeze(), img.squeeze(),num.squeeze(), 

    def __len__(self):
        return len(self.h5f['name'])

    
    def _get_item_from_keys(self, keys, idx):
        # for k in keys:
        res = np.stack([self.h5f[k][idx] for k in keys],-1)
        return res

    @staticmethod
    def read_h5(path_list):
        pass
    def __del__(self):
        self.h5f.close()






if __name__=='__main__': 
    with h5py.File("/mnt/d12t/mpi/Data/fold0.h5") as f:
        
        DATASET = mpi_dataset(f)
        dataloaders = torch.utils.data.DataLoader(
                    dataset=DATASET,
                    batch_size=5,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False,
                    drop_last=False)
        for f,l in dataloaders:
            a,b = f[0],f[1]
            print(":",a.shape,b.shape,l.shape)


# class DataFromH5File(data.Dataset):
#     def __init__(self, filepath):
#         h5File = h5py.File(filepath, 'r')
#         self.hr = h5File['hr']
#         self.lr = h5File['lr']
        
#     def __getitem__(self, idx):
#         label = torch.from_numpy(self.hr[idx]).float()
#         data = torch.from_numpy(self.lr[idx]).float()
#         return data, label
    
#     def __len__(self):
#         assert self.hr.shape[0] == self.lr.shape[0], "Wrong data length"
#         return self.hr.shape[0]

'''
self.labels = [
            "poi_num","slope","DHSCLUST1","LON","MPI_Easy4_fixed","Living Standards",
            "osm","pr_sum","NIR","TEMP1","GREEN","burnedCount","NIGHTLIGHTS","tmm_sum",
            "SWIR2","famale","AOD047","area","pr","AOD055","EVI","LAT","RED","age_struct_child",
            "MPI3","age_struct_young","conflict_num","tmmx","MPI3_fixed","year","road_length",
            "A","male","lat","lon","water","NDVI","age_struct_middle","aspect","place_num",
            "education","Individual_num","MPI_Easy4","SWIR1","death_num","Number of Households2",
            "health","tmmn","H","elevation","age_struct_old","viirs_v2","country",
            "Households_num","BLUE","building_area"
        ]


A (448, 1)
AOD047 (448, 255, 255)
AOD055 (448, 255, 255)
BLUE (448, 255, 255)
DHSCLUST1 (448, 1)
EVI (448, 255, 255)
GREEN (448, 255, 255)
H (448, 1)
Households_num (448, 1)
Individual_num (448, 1)
LAT (448, 255, 255)
LON (448, 255, 255)
Living Standards (448, 1)
MPI3 (448, 1)
MPI3_fixed (448, 1)
MPI_Easy4 (448, 1)
MPI_Easy4_fixed (448, 1)
NDVI (448, 255, 255)
NIGHTLIGHTS (448, 255, 255)
NIR (448, 255, 255)
Number of Households2 (448, 1)
RED (448, 255, 255)
SWIR1 (448, 255, 255)
SWIR2 (448, 255, 255)
TEMP1 (448, 255, 255)
age_struct_child (448, 255, 255)
age_struct_middle (448, 255, 255)
age_struct_old (448, 255, 255)
age_struct_young (448, 255, 255)
area (448, 1)
aspect (448, 255, 255)
building_area (448, 1)
burnedCount (448, 1)
conflict_num (448, 1)
country (448, 7)
death_num (448, 1)
education (448, 1)
elevation (448, 255, 255)
famale (448, 255, 255)
health (448, 1)
lat (448, 1)
lon (448, 1)
male (448, 255, 255)
name (448,)
osm (448, 255, 255)
place_num (448, 1)
poi_num (448, 1)
pr (448, 255, 255)
pr_sum (448, 1)
road_length (448, 1)
slope (448, 255, 255)
tmm_sum (448, 1)
tmmn (448, 255, 255)
tmmx (448, 255, 255)
viirs_v2 (448, 255, 255)
water (448, 1)
year (448, 1)
'''