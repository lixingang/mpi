import glob,os
from datetime import datetime
train_config = {
    "model_name":"mpi",
    "device":"cuda:0",
    "epochs":500,
    "batch_size":80,
    "log_dir":os.path.join('Logs',datetime.now().strftime('%b%d_%H-%M-%S')),
    "init_lr":0.001,
    "best_acc":{"name":"r2", "value":0.5}, #最佳权重精度的初始值init
    "restore_weight":None, #path
    "best_weight":None, #path
    # "h5_dir": glob.glob("/root/*fold0*.h5"),
    "h5_dir": glob.glob("/mnt/d12t/mpi/Data/*fold0?.h5"),
    "seed":0,
    

    "img_keys":["EVI","NDVI","NIGHTLIGHTS"],
    "num_keys":[
            "tmm_sum","Households_num","Individual_num","Number of Households2",
            "area","death_num","building_area","burnedCount","conflict_num",
            "death_num","place_num","poi_num","pr_sum","water","DHSCLUST1"
        ],
    "pos_keys":["lat","lon","name"],
    "label_keys":["MPI3_fixed"],
    "other_label_keys":["H","A","education","health","Living Standards"],
}







'''
"img_keys":["RED","BLUE","GREEN","SWIR2","EVI","NIR","NDVI","SWIR1","NIGHTLIGHTS","elevation"],
"num_keys":[
        "tmm_sum","Households_num","Individual_num","Number of Households2","Living Standards",
        "area","death_num","building_area","burnedCount","conflict_num","death_num","education",
        "health","place_num","poi_num","pr_sum","water","H","DHSCLUST1"
    ],
"pos_keys":["lat","lon","name"],
"label_keys":["MPI3_fixed"],
}
'''