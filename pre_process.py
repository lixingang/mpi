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


class PreProcess(object):
    def __init__(self):
        self.args = parse_yaml("Config/config.yaml")
        self.NORM_MIN = {
            "conflict_num": 0.0,
            "tmm_sum": 2550.8984375,
            "year": 2013.0,
            "H": 0.0,
            "place_num": 0.0,
            "death_num": 0.0,
            "SWIR1": -0.0012000000569969416,
            "age_struct_child": -13.785914421081543,
            "NIGHTLIGHTS": -0.03066941723227501,
            "BLUE": -0.16220000386238098,
            "MPI_Easy4_fixed": 0.0,
            "osm": -3.4028230607370965e38,
            "elevation": -59.0,
            "aspect": -0.0,
            "slope": 0.0,
            "EVI": -1888.0,
            "DHSCLUST1": 1.0,
            "building_area": 0.0,
            "health": 0.0,
            "age_struct_old": -0.6372672915458679,
            "GREEN": -0.07109999656677246,
            "pr_sum": 351.51861572265625,
            "Living Standards": 0.0,
            "TEMP1": 0.0,
            "A": 0.0,
            "lat": 4.290337085723877,
            "NDVI": -1958.0,
            "tmmn": 0.0,
            "age_struct_middle": -16.932937622070312,
            "AOD047": 0.0,
            "tmmx": 0.0,
            "Households_num": 4,
            "burnedCount": 0.0,
            "LAT": 4.256115913391113,
            "MPI_Easy4": 0.0,
            "MPI3": 0.0,
            "water": 0.0,
            "SWIR2": -0.007300000172108412,
            "poi_num": 0.0,
            "lon": 2.7638161182403564,
            "road_length": 0.0,
            "Individual_num": 11,
            "AOD055": 0.0,
            "LON": 2.7295758724212646,
            "RED": -0.04320000112056732,
            "education": 0.0,
            "pr": 0.0,
            "area": 0,
            "Lai": 0.0,
            "age_struct_young": -10.55878734588623,
            "famale": -19.354753494262695,
            "male": -24.56015396118164,
            "NIR": 0.0,
            "Number of Households2": 11.0,
            "MPI3_fixed": 0.0,
            "viirs_v2": -54.89772415161133,
            "DHSCLUST": 1.0,
            "Number of Households": 4.0,
            "Number of Individual": 11.0,
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
        self.NORM_MAX = {
            "conflict_num": 23.0,
            "tmm_sum": 3563.697509765625,
            "year": 2018.0,
            "H": 1.0,
            "place_num": 319.0,
            "death_num": 84.0,
            "SWIR1": 2.0,
            "age_struct_child": 239.66905212402344,
            "NIGHTLIGHTS": 2944.239990234375,
            "BLUE": 2.0,
            "MPI_Easy4_fixed": 0.22499999403953552,
            "osm": 2338.15283203125,
            "elevation": 1781.0,
            "aspect": 359.7214050292969,
            "slope": 79.88656616210938,
            "EVI": 7472.0,
            "DHSCLUST1": 1400.0,
            "building_area": 0.0009877159027382731,
            "health": 0.8659999966621399,
            "age_struct_old": 39.81295394897461,
            "GREEN": 1.195449948310852,
            "pr_sum": 3856.78662109375,
            "Living Standards": 1.0,
            "TEMP1": 315.54998779296875,
            "A": 0.8330000042915344,
            "lat": 13.767678260803223,
            "NDVI": 9094.0,
            "tmmn": 248.5833282470703,
            "age_struct_middle": 527.4027709960938,
            "AOD047": 1750.3333740234375,
            "tmmx": 369.75,
            "Households_num": 45,
            "burnedCount": 224.0,
            "LAT": 13.800820350646973,
            "MPI_Easy4": 0.22499999403953552,
            "MPI3": 0.828000009059906,
            "water": 29621.0,
            "SWIR2": 2.0,
            "poi_num": 1704.0,
            "lon": 13.932452201843262,
            "road_length": 742.8068237304688,
            "Individual_num": 373,
            "AOD055": 1323.3333740234375,
            "LON": 13.966691970825195,
            "RED": 2.0,
            "education": 1.0,
            "pr": 328.75,
            "area": 1,
            "Lai": 42.30769348144531,
            "age_struct_young": 223.69056701660156,
            "famale": 495.7632141113281,
            "male": 532.8121948242188,
            "NIR": 1.3014999628067017,
            "Number of Households2": 200.0,
            "MPI3_fixed": 0.828000009059906,
            "viirs_v2": 3889.79443359375,
            "DHSCLUST": 1400.0,
            "Number of Households": 45.0,
            "Number of Individual": 373.0,
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
        self.not_norm_list = [
            "Country",
            "country",
            "year",
            "LAT",
            "LON",
            "lat",
            "lon",
            "name",
            "H",
            "A",
            "education",
            "health",
            "MPI3_fixed",
            "Living Standards",
            "MPI_Easy4_fixed",
            "DHSCLUST1",
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
            "Number of Households",
            "Number of Individual",
        ]

    def tf2pth_origin(self, source_dir="Data/raw_data", target_dir="Data/origin"):

        os.makedirs(target_dir, exist_ok=True)
        mpi_indicator = pd.read_csv("Data/nga_mpi.csv", low_memory=False)
        for f in tqdm(os.listdir(source_dir)):
            it = tf.compat.v1.python_io.tf_record_iterator(os.path.join(source_dir, f))

            content = decode_example(next(it))
            res = {}
            for key in content.keys():
                if isinstance(content[key], list):
                    continue

                if key in self.args["D"]["origin_img_keys"]:
                    res[key] = np.reshape(content[key], (255, 255))[16:240, 16:240]

                if key in self.args["D"]["num_keys"]:
                    res[key] = content[key]

                if key in self.args["D"]["label_keys"]:
                    res[key] = content[key]

            # 在pth文件中加入nga_mpi中的额外信息
            dhsclust = int(content["DHSCLUST1"].item())
            year = int(content["year"].item())

            search_result = mpi_indicator.loc[
                (mpi_indicator["DHSCLUST"] == dhsclust)
                & (mpi_indicator["Year"] == year)
            ].head()

            for (col_name, col_data) in search_result.iteritems():
                res[col_name] = np.asarray([col_data.item()])

            save_name = f"nga_{year}_{dhsclust}.pth"
            torch.save(
                res,
                os.path.join(target_dir, save_name),
                _use_new_zipfile_serialization=False,
            )

        # self._get_info(target_dir)

    def tf2pth_reduce(self, source_dir="Data/raw_data", target_dir="Data/input_data2"):

        os.makedirs(target_dir, exist_ok=True)
        mpi_indicator = pd.read_csv("Data/nga_mpi.csv", low_memory=False)
        for f in tqdm(os.listdir(source_dir)):
            it = tf.compat.v1.python_io.tf_record_iterator(os.path.join(source_dir, f))

            content = decode_example(next(it))
            res = {}
            for key in content.keys():
                if isinstance(content[key], list):
                    continue
                # if key in ['']
                # if key in ["male", "famale"]:
                #     sex = (content["famale"] - content["male"]) / (
                #         content["famale"] + content["male"]
                #     )

                #     pop = content["famale"] + content["male"]
                #     res["sex"] = np.reshape(sex, (255, 255))[16:240, 16:240]
                #     res["pop"] = np.reshape(pop, (255, 255))[16:240, 16:240]

                # if key in [
                #     "age_struct_child",
                #     "age_struct_young",
                #     "age_struct_middle",
                #     "age_struct_old",
                # ]:
                #     total_pop = (
                #         content["age_struct_child"]
                #         + content["age_struct_young"]
                #         + content["age_struct_middle"]
                #         + content["age_struct_old"]
                #     )
                #     specical_pop = (
                #         content["age_struct_child"] + content["age_struct_old"]
                #     )
                #     age = specical_pop / total_pop

                #     res["age"] = np.reshape(age, (255, 255))[16:240, 16:240]

                if key in self.args["D"]["img_keys"]:
                    res[key] = np.reshape(content[key], (255, 255))[16:240, 16:240]

                if key in self.args["D"]["num_keys"]:
                    res[key] = content[key]

                if key in self.args["D"]["label_keys"]:
                    res[key] = content[key]
                # if key in self.args["D"]["indicator_keys"]:
                #     res[key] = content[key]

                # if res[key].shape[0] == 255 * 255:
                #     # res[key] = np.reshape(f[key], (255, 255))
                #     hist, bins = np.histogram(
                #         res[key],
                #         bins=20,
                #         range=(self.NORM_MIN[key], self.NORM_MAX[key]),
                #         density=True,
                #     )
                #     if np.isnan(np.sum(hist)):
                #         print("replaced by zeros")
                #         hist = np.zeros_like(hist)
                #     res[key] = hist

                # if key not in self.not_norm_list:
                #     res[key] = (res[key] - self.NORM_MIN[key]) / (
                #         self.NORM_MAX[key] - self.NORM_MIN[key]
                #     )

            # 在pth文件中加入nga_mpi中的额外信息
            dhsclust = int(content["DHSCLUST1"].item())
            year = int(content["year"].item())

            search_result = mpi_indicator.loc[
                (mpi_indicator["DHSCLUST"] == dhsclust)
                & (mpi_indicator["Year"] == year)
            ].head()

            for (col_name, col_data) in search_result.iteritems():
                res[col_name] = np.asarray([col_data.item()])

            save_name = f"nga_{year}_{dhsclust}.pth"
            torch.save(
                res,
                os.path.join(target_dir, save_name),
                _use_new_zipfile_serialization=False,
            )

        self.get_info(target_dir)

    def get_info(self, source_dir="Data/origin"):
        label_list = {"name": []}
        mpi_indicator = [
            "MPI3_fixed",
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

    def inspect_pth(self, pth_dir="Data/origin"):
        for file in os.listdir(pth_dir):
            f = torch.load(os.path.join(pth_dir, file))
            for key in f.keys():
                if isinstance(f[key], (str, float)):
                    print(key, f[key])
                else:
                    print(key, f[key].shape)

            break

    def inspect_tf(self, tf_dir="Data/raw_data"):
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

    def get_norm_parameters(self, pth_dir="pth"):
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
                    min_value = np.min(f[key]).item()
                    NORM_MIN[key] = (
                        min_value if min_value < NORM_MIN[key] else NORM_MIN[key]
                    )
                    max_value = np.max(f[key]).item()
                    NORM_MAX[key] = (
                        max_value if max_value > NORM_MAX[key] else NORM_MAX[key]
                    )

        print(NORM_MIN)
        print(NORM_MAX)


if __name__ == "__main__":
    fire.Fire(PreProcess)
    # pass
