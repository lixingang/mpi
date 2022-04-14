import torch
import h5py
import numpy as np
import os,sys,glob
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
        name = self.h5f[self.pos_keys[-1]][i]
        return label.squeeze(), img.squeeze(),num.squeeze(), name

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







