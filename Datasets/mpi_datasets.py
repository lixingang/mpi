import torch
import h5py
import numpy as np
import os,sys,glob
# from config import train_config
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

    # import office packages
    import os,sys,logging,argparse,h5py,glob,time,random,datetime
    sys.path.append(os.getcwd())
    print(sys.path)
    import torch
    from torch.utils.data import DataLoader,ConcatDataset
    import numpy as np
    # import in-project packages
    from Losses.loss import HEMLoss,CenterLoss
    from Models.network import Net
    from Datasets.mpi_datasets import mpi_dataset
    from Utils.AverageMeter import AverageMeter
    from Utils.clock import clock,Timer
    from Utils.setup_seed import setup_seed
    from Utils.ParseYAML import ParseYAML

    config = ParseYAML("config.yaml")
    config['log_dir'] = os.path.join("Logs", config["model_name"], datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
    parser = argparse.ArgumentParser(description='Process some integers.')
    for key in config:
        parser.add_argument(f'--{key}', default=config[key],type=type(config[key]))
    parser.add_argument(f'--note', default="",type=str)
    args = parser.parse_args()
    setup_seed(args.seed)

    ds = ConcatDataset([mpi_dataset(args, h5path) for h5path in args.h5_dir])

    train_size = int(len(ds) * 0.6)
    validate_size = int(len(ds) * 0.2)
    test_size = len(ds) - validate_size - train_size
    print(f"[Data] train,validate,test size: {train_size},{validate_size},{test_size}")

    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, validate_size, test_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=6,
        pin_memory=True,
        drop_last=False,
    )
    validate_loader = DataLoader(
        validate_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=False,
    )

    for y, fea_img, fea_num in train_loader:
        print(y.shape, fea_img.shape, fea_num.shape)







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
BLUE (448, 255, 255) √
DHSCLUST1 (448, 1)
EVI (448, 255, 255) √
GREEN (448, 255, 255) √
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
NDVI (448, 255, 255) √
NIGHTLIGHTS (448, 255, 255) √
NIR (448, 255, 255)
Number of Households2 (448, 1)
RED (448, 255, 255) √
SWIR1 (448, 255, 255) √
SWIR2 (448, 255, 255) √
TEMP1 (448, 255, 255) √
age_struct_child (448, 255, 255) √
age_struct_middle (448, 255, 255) √
age_struct_old (448, 255, 255) √
age_struct_young (448, 255, 255) √
area (448, 1)
aspect (448, 255, 255) √
building_area (448, 1)
burnedCount (448, 1)
conflict_num (448, 1)
country (448, 7)
death_num (448, 1)
education (448, 1)
elevation (448, 255, 255) √
famale (448, 255, 255) √
health (448, 1)
lat (448, 1)
lon (448, 1)
male (448, 255, 255) √
name (448,)
osm (448, 255, 255) √
place_num (448, 1)
poi_num (448, 1)
pr (448, 255, 255) √
pr_sum (448, 1)
road_length (448, 1)
slope (448, 255, 255) √
tmm_sum (448, 1)
tmmn (448, 255, 255) √
tmmx (448, 255, 255) √
viirs_v2 (448, 255, 255) √
water (448, 1)
year (448, 1)
'''