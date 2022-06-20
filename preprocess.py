import tensorflow as tf
import numpy as np
from tfrecord_lite import decode_example
import torch
import os
from tqdm import tqdm
import pandas as pd
import fire
import glob
from Utils.base import parse_yaml


args = parse_yaml("swint.yaml")
NORM_MIN = {
    "MPI3_fixed": 0.0,
    "lat": 4.2903,
    "age_struct_child": -13.785914421081543,
    "building_area": 0.0,
    "lon": 2.7638,
    "male": -24.56015396118164,
    "age_struct_middle": -16.932937622070312,
    "place_num": 0.0,
    "age_struct_young": -10.55878734588623,
    "burnedCount": 0.0,
    "GREEN": -0.07109999656677246,
    "SWIR2": -0.007300000172108412,
    "BLUE": -0.16220000386238098,
    "tmmx": 0.0,
    "elevation": -59.0,
    "conflict_num": 0.0,
    "pr": 0.0,
    "NIGHTLIGHTS": -0.03066941723227501,
    "viirs_v2": -54.89772415161133,
    "road_length": 0.0,
    "death_num": 0.0,
    "age_struct_old": -0.6372672915458679,
    "famale": -19.354753494262695,
    "NIR": 0.0,
    "year": 2013.0,
    "water": 0.0,
    "TEMP1": 0.0,
    "Households_num": 4,
    "RED": -0.04320000112056732,
    "tmmn": 0.0,
    "poi_num": 0.0,
    "SWIR1": 0.0,
    "DHSCLUST": 1,
    "Year": 2013,
    "Child mortality": 0.0,
    "Nutrition": 0.0,
    "School attendance": 0.0,
    "Years of schooling": 0.0,
    "Electricity": 0.0,
    "Drinking water": 0.0,
    "Sanitation": 0.0,
    "Housing": 0.0,
    "Cooking fuel": 0.0,
    "Assets": 0.0,
}
NORM_MAX = {
    "MPI3_fixed": 0.828,
    "lat": 13.7677,
    "age_struct_child": 239.66905212402344,
    "building_area": 0.0009877159027382731,
    "lon": 13.9325,
    "male": 532.8121948242188,
    "age_struct_middle": 527.4027709960938,
    "place_num": 319.0,
    "age_struct_young": 223.69056701660156,
    "burnedCount": 224.0,
    "GREEN": 1.195449948310852,
    "SWIR2": 2.0,
    "BLUE": 2.0,
    "tmmx": 369.75,
    "elevation": 1781.0,
    "conflict_num": 23.0,
    "pr": 328.75,
    "NIGHTLIGHTS": 2944.239990234375,
    "viirs_v2": 3889.79443359375,
    "road_length": 742.8068237304688,
    "death_num": 84.0,
    "age_struct_old": 39.81295394897461,
    "famale": 495.7632141113281,
    "NIR": 1.152250051498413,
    "year": 2018.0,
    "water": 29621.0,
    "TEMP1": 315.5,
    "Households_num": 45,
    "RED": 2.0,
    "tmmn": 248.5833282470703,
    "poi_num": 1704.0,
    "SWIR1": 2.0,
    "DHSCLUST": 1400,
    "Year": 2018,
    "Child mortality": 0.78,
    "Nutrition": 1.0,
    "School attendance": 1.0,
    "Years of schooling": 1.0,
    "Electricity": 1.0,
    "Drinking water": 1.0,
    "Sanitation": 1.0,
    "Housing": 1.0,
    "Cooking fuel": 1.0,
    "Assets": 1.0,
}
not_norm_list = [
    "year",
    "lat",
    "lon",
    "MPI3_fixed",
    "Assets",
    "Cooking fuel",
    "Housing",
    "Sanitation",
    "Drinking water",
    "Electricity",
    "Years of schooling",
    "School attendance",
    "Nutrition",
    "Child mortality",
]

# [31:  223, 31:  223]
# [15:  239,15:  239]
def tf2pth(source_dir="Data/raw_data", target_dir="Data/NROM224", t_type=np.float32):

    os.makedirs(target_dir, exist_ok=True)
    mpi_indicator = pd.read_csv("Data/nga_mpi.csv", low_memory=False)
    for f in tqdm(os.listdir(source_dir)):
        it = tf.compat.v1.python_io.tf_record_iterator(os.path.join(source_dir, f))

        content = decode_example(next(it))

        res = {}
        for key in content.keys():
            if isinstance(content[key], list):
                continue

            if key in args.D.img_keys:
                res[key] = np.reshape(content[key], (255, 255))[15:239, 15:239]
                res[key] = (
                    (res[key] - NORM_MIN[key]) / (NORM_MAX[key] - NORM_MIN[key]) * 255
                ).astype(t_type)
                res[key] = torch.from_numpy(res[key])

            if key in args.D.num_keys:
                res[key] = content[key]
                res[key] = (
                    (res[key] - NORM_MIN[key]) / (NORM_MAX[key] - NORM_MIN[key]) * 255
                ).astype(t_type)
                res[key] = torch.from_numpy(res[key])
            if key in args.D.label_keys:
                res[key] = content[key]
                res[key] = torch.from_numpy(res[key])
                # if key == "MPI3_fixed":
                #     res[key] = np.where(res[key] > 0.7, 0.71, res[key])

        # 在pth文件中加入nga_mpi中的额外信息
        dhsclust = int(content["DHSCLUST1"].item())
        year = int(content["year"].item())

        search_result = mpi_indicator.loc[
            (mpi_indicator["DHSCLUST"] == dhsclust) & (mpi_indicator["Year"] == year)
        ].head()

        for (col_name, col_data) in search_result.iteritems():
            if isinstance(col_data.item(), str):
                continue
            res[col_name] = torch.from_numpy(np.asarray([col_data.item()]))

        save_name = f"nga_{year}_{dhsclust}.pth"
        # if res["MPI3_fixed"] <= 0.001:
        #     continue
        torch.save(
            res,
            os.path.join(target_dir, save_name),
            _use_new_zipfile_serialization=False,
        )
    # _get_info(target_dir)


# def tf2pth(source_dir="Data/raw_data", target_dir="Data/origin224"):

#     os.makedirs(target_dir, exist_ok=True)
#     mpi_indicator = pd.read_csv("Data/nga_mpi.csv", low_memory=False)
#     for f in tqdm(os.listdir(source_dir)):
#         it = tf.compat.v1.python_io.tf_record_iterator(os.path.join(source_dir, f))

#         content = decode_example(next(it))
#         img_array = []
#         res = {}
#         for key in content.keys():
#             if isinstance(content[key], list):
#                 continue

#             if key in args.D.img_keys:
#                 img_array.append(np.reshape(content[key], (255, 255))[15:239, 15:239])
#             if key in args.D.num_keys:
#                 res[key] = content[key]
#             if key in args.D.label_keys:
#                 res[key] = content[key]

#         img_array = np.stack(img_array, 0)
#         res["img"] = img_array
#         # 在pth文件中加入nga_mpi中的额外信息
#         dhsclust = int(content["DHSCLUST1"].item())
#         year = int(content["year"].item())

#         search_result = mpi_indicator.loc[
#             (mpi_indicator["DHSCLUST"] == dhsclust) & (mpi_indicator["Year"] == year)
#         ].head()

#         for (col_name, col_data) in search_result.iteritems():
#             if isinstance(col_data.item(), str):
#                 continue
#             res[col_name] = torch.from_numpy(np.asarray([col_data.item()]))

#         save_name = f"nga_{year}_{dhsclust}.pth"
#         torch.save(
#             res,
#             os.path.join(target_dir, save_name),
#             _use_new_zipfile_serialization=False,
#         )


def get_info(source_dir="Data/origin"):
    label_list = {"name": []}
    mpi_indicator = [
        "MPI3_fixed",
        "DHSCLUST",
        "Year",
        "lon",
        "lat",
        "Child mortality",
        "Nutrition",
        "School attendance",
        "Years of schooling",
        "Electricity",
        "Drinking water",
        "Sanitation",
        "Housing",
        "Cooking fuel",
        "Assets",
    ]

    for i in tqdm(glob.glob(f"{source_dir}/*.pth")):
        label_list["name"].append(os.path.basename(i))

        for ind in mpi_indicator:
            if ind not in label_list.keys():
                label_list[ind] = []
            label_list[ind].append(round(torch.load(i)[ind].item(), 4))
    df = pd.DataFrame(label_list)
    df.to_csv("data.csv", index=None)


def inspect_pth(pth_dir="Data/origin224"):
    for file in os.listdir(pth_dir):
        f = torch.load(os.path.join(pth_dir, file))
        for key in f.keys():
            if key == "poi_num":
                print(f[key])
            if isinstance(f[key], (str, float)):
                print(key, f[key])
            else:
                print(key, f[key].shape)

        break


def inspect_tf(tf_dir="Data/raw_data"):
    file = os.path.join(tf_dir, os.listdir(tf_dir)[0])
    it = tf.compat.v1.python_io.tf_record_iterator(file)
    content = decode_example(next(it))
    for key in content.keys():
        if isinstance(content[key], list):
            print(key, len(content[key]))
        elif isinstance(content[key], np.ndarray):
            print(key, content[key].shape)
        else:
            pass


def get_norm_parameters(pth_dir="Data/origin224"):
    NORM_MIN = {}
    NORM_MAX = {}
    for file in tqdm(os.listdir(pth_dir)):
        f = torch.load(os.path.join(pth_dir, file))
        for key in f.keys():
            if key == "Country":
                continue
            if key not in NORM_MIN.keys():
                NORM_MIN[key] = 999999
                NORM_MAX[key] = -999999

            else:

                min_value = f[key].min().item()
                NORM_MIN[key] = (
                    min_value if min_value < NORM_MIN[key] else NORM_MIN[key]
                )
                max_value = f[key].max().item()
                NORM_MAX[key] = (
                    max_value if max_value > NORM_MAX[key] else NORM_MAX[key]
                )

    print(NORM_MIN)
    print(NORM_MAX)


if __name__ == "__main__":
    fire.Fire()
    # pass
