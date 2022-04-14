##### 从raw_data中挑选真值>0.00001的保存为raw_data_removed
import tfrecord as tfr
import h5py
import os,sys
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import shutil

import tfrecord as tfr
import h5py
import os,sys
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm

norm_max = {
    'poi_num': 420.0, 'slope': 79.886566, 'DHSCLUST1': 1311.0, 'LON': 13.966692, 'MPI_Easy4_fixed': 0.207, 'Living Standards': 1.0, 
    'osm': 2338.1528, 'pr_sum': 3586.076, 'NIR': 1.3015, 'TEMP1': 315.55, 'GREEN': 1.0481, 'burnedCount': 224.0, 'NIGHTLIGHTS': 2944.24, 
    'tmm_sum': 3563.6975, 'SWIR2': 2.0, 'famale': 495.7632, 'AOD047': 1750.3334, 'area': 1, 'pr': 301.0, 'AOD055': 1323.3334, 'EVI': 7472.0, 
    'LAT': 13.80082, 'RED': 2.0, 'age_struct_child': 239.66905, 'MPI3': 0.828, 'age_struct_young': 223.69057, 'conflict_num': 23.0, 'tmmx': 369.75, 
    'MPI3_fixed': 0.828, 'year': 2018.0, 'road_length': 675.15234, 'A': 0.833, 'male': 532.8122, 'lat': 13.767678, 'lon': 13.932452, 'water': 29621.0, 
    'NDVI': 9094.0, 'age_struct_middle': 527.4028, 'aspect': 359.7214, 'place_num': 319.0, 'education': 1.0, 'Individual_num': 373, 'MPI_Easy4': 0.207, 
    'SWIR1': 2.0, 'death_num': 46.0, 'Number of Households2': 200.0, 'health': 0.866, 'tmmn': 248.58333, 'H': 1.0, 'elevation': 1781.0, 
    'age_struct_old': 39.812954, 'viirs_v2': 3889.7944, 'Households_num': 45, 'BLUE': 1.03855, 'building_area': 0.0009798642, 'country': 114
    }
norm_min = {
    'poi_num': 0.0, 'slope': 0.0, 'DHSCLUST1': 1.0, 'LON': 2.7295759, 'MPI_Easy4_fixed': 0.0, 'Living Standards': 0.0039, 'osm': -3.402823e+38, 
    'pr_sum': 351.51862, 'NIR': 0.0, 'TEMP1': 0.0, 'GREEN': -0.0082, 'burnedCount': 0.0, 'NIGHTLIGHTS': -0.030669417, 'tmm_sum': 2550.8984, 
    'SWIR2': -0.0027, 'famale': -19.354753, 'AOD047': 0.0, 'area': 0, 'pr': 0.0, 'AOD055': 0.0, 'EVI': -1888.0, 'LAT': 4.256116, 'RED': -0.01395, 
    'age_struct_child': -13.785914, 'MPI3': 0.0, 'age_struct_young': -10.558787, 'conflict_num': 0.0, 'tmmx': 0.0, 'MPI3_fixed': 0.0018, 'year': 2013.0, 
    'road_length': 0.0, 'A': 0.0634, 'male': -24.560154, 'lat': 4.290337, 'lon': 2.763816, 'water': 0.0, 'NDVI': -1958.0, 'age_struct_middle': -16.932938, 
    'aspect': -0.0, 'place_num': 0.0, 'education': 0.0, 'Individual_num': 11, 'MPI_Easy4': 0.0, 'SWIR1': -0.0012, 'death_num': 0.0, 
    'Number of Households2': 11.0, 'health': 0.0, 'tmmn': 0.0, 'H': 0.0238, 'elevation': -59.0, 'age_struct_old': -0.6372673, 'viirs_v2': -38.779884, 
    'Households_num': 4, 'BLUE': -0.1329, 'building_area': 0.0, 'country': 97}

extra_info = {}
class TfrecordWorker():
    def __init__(self,tfr_list):
        
        self.info = {"label":[],"typee":[],"shape":[]}
        self.data_dir = "raw_data_removed"
        self.tfr_list = tfr_list
        self.tfr_description = self._parse_description("label_type.csv")
        loader = tfr.tfrecord_loader(self.tfr_list[0], None, self.tfr_description  )
        for record in loader:
            for key in record.keys():
                self.info['label'].append(key)
                self.info['typee'].append(type(record[key][0]))
                self.info['shape'].append(record[key].shape)
        self.attr_size = len(self.info['label'])
        self.data_size = len(self.tfr_list)
        print(f"总共有{self.attr_size}个属性")
        print(f"总共有{self.data_size}个tfrecord文件")


    def create_h5f(self, h5path="./data.h5"):
        self.h5f = h5py.File(h5path, 'w')
        self.dset = {}
        for i in range(self.attr_size):
            label = self.info["label"][i]
            typee = self.info["typee"][i]
            shape = self.info["shape"][i]
            if shape[0]==255*255:
                self.dset[label] = self.h5f.create_dataset(
                    label,
                    shape=[self.data_size,255*255],
                    compression=None,
                    dtype=typee 
                )
            else:
                self.dset[label] = self.h5f.create_dataset(label,
                                shape=[self.data_size, shape[0]],
                                compression=None,
                                dtype=typee)
                                
        
        self.dset["name"] = self.h5f.create_dataset("name",
                            shape=[self.data_size],
                            compression=None,
                            dtype=h5py.special_dtype(vlen=str))
    def write_h5f(self):
        for idx,tfr_path in tqdm(enumerate(self.tfr_list)):
            self._write_one_item(tfr_path, idx)
            # if idx>2:
            #     break
            
        
    def close_h5f(self):
        self.h5f.close()

    def _write_one_item(self, tfr_path, idx):
        loader = tfr.tfrecord_loader(tfr_path, None, self.tfr_description  )
        for record in loader:
            # print(record['country'])
            for key in record.keys(): 

                if key not in extra_info.keys():
                    extra_info[key]=[]

                content = record[key]
                if content.shape[0]==255*255:
                    # content = np.reshape(content, (255,255))
                    content = content

                if key not in ["year","LAT","LON","lat","lon","name","H","A","education","health","Living Standards"]:
                    content = (content-norm_min[key]) / (norm_max[key]-norm_min[key])



                self.dset[key][idx] = content

                
                # extra_info[key].append([
                #         np.max(content),
                #         np.min(content),
                #     ])
        self.dset["name"][idx] = tfr_path.split("/")[-1]

    def _parse_description(self, csv_path):
        label_type = pd.read_csv(csv_path, usecols=["label","type"])
        description = {}
        for _, row in label_type.iterrows():
            description[str(row['label']).strip()] = str(row['type']).strip()
        return description



def start(files, savename):
    worker = TfrecordWorker(files)
    worker.create_h5f(savename)
    worker.write_h5f()
    worker.close_h5f()

# start(glob.glob("raw_data/*.tfrecord"),"data.h5")
start(glob.glob("raw_data_removed/*fold0*.tfrecord"),"v2/subsetfold0.h5")
start(glob.glob("raw_data_removed/*fold1*.tfrecord"),"v2/subsetfold1.h5")
start(glob.glob("raw_data_removed/*fold2*.tfrecord"),"v2/subsetfold2.h5")
start(glob.glob("raw_data_removed/*fold3*.tfrecord"),"v2/subsetfold3.h5")
# start(glob.glob("raw_data/*fold0*.tfrecord"),"test.h5")

# for key in extra_info.keys():
#     print(key)
#     print(extra_info[key])


