# import office packages
import enum
import os
import sys
import yaml
import fire
import torch
import logging
import argparse
import glob
import time
import random
import datetime
import shutil
import math
import copy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torchmetrics.functional import r2_score, mean_squared_error
from torchinfo import summary
import albumentations as A
import torch.multiprocessing as mp

# import in-project packages
from Losses.loss import *
from Losses.tri_loss import calc_triplet_loss
from Models import *
from Models.LDS import get_lds_weights
from Datasets.mpi_datasets import mpi_dataset
from Utils.base import parse_yaml, setup_seed, Meter, SaveOutput, split_train_valid
from Utils.predict_utils import *


def logging_setting(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"run.log"),
        filemode="w",
        format="%(asctime)s: %(message)s",
        # format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.DEBUG,
    )


# def prepare_folder(args):

#     if os.path.exists(args.M.current_log_dir):
#         message = input(
#             "[INFO] found duplicate directories, whether to overwrite (Y/N/R) "
#         )
#         if message.lower() == "y":
#             print("[INFO] deleting the existing logs...")
#             shutil.rmtree(args.M.current_log_dir)
#         elif message.lower() == "r":
#             print("[INFO] restoring the existing logs...")
#             return "restore"
#         else:
#             print("[INFO] canceled")
#             sys.exit(0)

#     os.makedirs(args.M.current_log_dir, exist_ok=True)


def prepare_datalist(args):
    """
    创建Logs/tags文件夹(不覆盖),
    生成包含train_list, valid_list, test_list的yaml文件
    """
    if not os.path.exists(args.M.parent_log_dir):
        os.mkdir(args.M.parent_log_dir)
    data_list = pd.read_csv(args.D.source)["name"].to_numpy()
    value_list = pd.read_csv(args.D.source)["MPI3_fixed"].to_numpy()
    sorted_id = sorted(
        range(len(value_list)), key=lambda k: value_list[k], reverse=False
    )
    data_list = data_list[sorted_id]
    value_list = value_list[sorted_id]

    assert args.M.split_method in ["cv", "holdout"], "split_method必须为cv, holdout中一种"

    """
    根据data_list与value_list, 生成不同划分方式下的train-, valid-, test-index
    """
    if args.M.split_method == "cv":
        # 以下为交叉验证的分割
        fold = {"name": [], "value": []}
        for i in range(args.M.k_fold):
            fold["name"].append(
                [v for v in range(0 + i, len(data_list), args.M.k_fold)]
            )
            fold["value"].append(
                [v for v in range(0 + i, len(value_list), args.M.k_fold)]
            )
        for i in range(args.M.k_fold):
            test_part = fold["name"][i]
            train_part = {"name": [], "value": []}
            for j in range(args.M.k_fold):
                if j == i:
                    continue
                else:
                    train_part["name"] = train_part["name"] + fold["name"][j]
                    train_part["value"] = train_part["value"] + fold["value"][j]
            train_index, valid_index = split_train_valid(
                np.array(train_part["name"]), [0.75, 0.25]
            )
            test_index = np.array(test_part)
            yamlname = os.path.join(args.M.parent_log_dir, f"{i}.yaml")
            print(f"[INFO] creating C-V record: {yamlname}")
            yamlname = os.path.join(args.M.parent_log_dir, str(i + 1)) + ".yaml"
            with open(yamlname, "w") as f:
                yaml.dump(
                    {
                        "log_dir": args.M.parent_log_dir,
                        "train_list": data_list[train_index].tolist(),
                        "valid_list": data_list[valid_index].tolist(),
                        "test_list": data_list[test_index].tolist(),
                        "train_value": value_list[train_index].tolist(),
                        "valid_value": value_list[valid_index].tolist(),
                        "test_value": value_list[test_index].tolist(),
                    },
                    f,
                )

    elif args.M.split_method == "holdout":
        min_value = 0.0
        max_value = 0.7
        step = 0.1
        iter_list = np.arange(min_value, max_value, step).tolist() + [1.0]
        hist_index = {}
        for i in range(len(iter_list) - 1):
            start = iter_list[i]
            end = iter_list[i + 1]
            hist_index[round(end, 2)] = np.where(
                (value_list >= start) & (value_list < end)
            )[0]
        count_of_hist = {k: len(hist_index[k]) for k in hist_index.keys()}
        num_test = {k: int(min(count_of_hist.values()) / 4) for k in hist_index.keys()}
        num_valid = {k: int(min(count_of_hist.values()) / 4) for k in hist_index.keys()}
        num_train = {
            k: count_of_hist[k] - int(min(count_of_hist.values()) / 4 * 2)
            for k in hist_index.keys()
        }
        for k in hist_index.keys():
            random.shuffle(hist_index[k])

        test_index = []
        valid_index = []
        train_index = []
        for k in hist_index.keys():
            test_index.extend(hist_index[k][: num_test[k]])
            valid_index.extend(hist_index[k][num_test[k] : num_test[k] + num_valid[k]])
            train_index.extend(hist_index[k][num_test[k] + num_valid[k] :])

        yamlname = os.path.join(args.M.parent_log_dir, "1.yaml")
        with open(yamlname, "w") as f:
            yaml.dump(
                {
                    "log_dir": args.M.parent_log_dir,
                    "train_list": data_list[train_index].tolist(),
                    "valid_list": data_list[valid_index].tolist(),
                    "test_list": data_list[test_index].tolist(),
                    "train_value": value_list[train_index].tolist(),
                    "valid_value": value_list[valid_index].tolist(),
                    "test_value": value_list[test_index].tolist(),
                },
                f,
            )
    else:
        raise NotImplementedError


def prepare_model(args):
    assert args.M.model.lower() in {"vit", "mlp", "swint"}
    model = None
    callback = {}

    # callback["weights_fea"] = SaveOutput()

    if args.M.model.lower() == "mlp":
        model = MLP(
            len(args.D.img_keys) * args.D.in_channel,
            len(args.D.num_keys),
        ).cuda()
        print(model, file=open(os.path.join(args.M.current_log_dir, "model.txt"), "a"))
        hook1 = model.head1[0].register_forward_hook(callback["neck_feat"])
        hook2 = model.fclayer[4].register_forward_hook(callback["weights_fea"])

    elif args.M.model.lower() == "swint":
        callback["neck_feat"] = SaveOutput()
        callback["img_fea"] = SaveOutput()
        callback["num_fea"] = SaveOutput()
        model = SwinTransformer(
            img_size=args.M.img_size,
            patch_size=4,
            in_chans=(len(args.D.img_keys), len(args.D.num_keys)),
            num_classes=256,
            embed_dim=96,
            depths=[2, 2, 6],
            num_heads=[3, 6, 12],
            window_size=int(args.M.img_size / 32),
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.00,
            drop_path_rate=0.01,
        ).cuda()

        # model_parameter = summary(
        #     model,
        #     input_size=[
        #         (
        #             1,
        #             len(args.D.img_keys),
        #             args.M.img_size,
        #             args.M.img_size,
        #         ),
        #         (1, len(args.D.num_keys)),
        #     ],
        #     verbose=0,
        # )
        # print(
        #     model_parameter,
        #     file=open(os.path.join(args.M.parent_log_dir, "model_parameter.txt"), "w"),
        # )

        hook1 = model.neck[-1].register_forward_hook(callback["neck_feat"])
        hook3 = model.avgpool.register_forward_hook(callback["img_fea"])
        hook4 = model.num_layers[-1].register_forward_hook(callback["num_fea"])

    return model, callback


def prepare_gpu(m_size=5200):
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >.tmp")
    memory_gpu = [int(x.split()[2]) for x in open(".tmp", "r").readlines()]
    while max(memory_gpu) < m_size:
        time.sleep(30)
        print("[INFO] Query the idle GPU ...")
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >.tmp")
        memory_gpu = [int(x.split()[2]) for x in open(".tmp", "r").readlines()]

    print(f"[INFO] GPU:{np.argmax(memory_gpu)} was selected")
    best_gpu = int(np.argmax(memory_gpu))
    return str(best_gpu)


def train_epoch(args, model, callback, loader, optimizer, writer, gp=None):

    model.train()
    epoch = args.M.crt_epoch
    meters = {
        "loss": Meter(),
        "y": Meter(),
        "yhat": Meter(),
        "loss1": Meter(),
        "loss2": Meter(),
        "tri_loss": Meter(),
    }
    for img, num, lbl, ind in loader:
        img = img.float().cuda()
        num = num.float().cuda()
        y = lbl["MPI3_fixed"].float().cuda()
        ind = ind.float().cuda()
        out = model(img, num)
        yhat, indhat, feat = out[0], out[1], out[2]

        loss = args.M.losses.loss * weighted_huber_loss(yhat, y)
        if "ind_loss" in args.M.losses.keys():
            loss2 = args.M.losses.ind_loss * weighted_huber_loss(indhat, num)
            loss += loss2
        if "tri_loss" in args.M.losses.keys():
            loss3 = args.M.losses.tri_loss * calc_triplet_loss(feat, y - 0, 0.8 - 0)
            loss += loss3
        # loss = loss1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        meters["loss"].update(loss)
        # meters["loss2"].update(loss2)
        meters["y"].update(y)
        meters["yhat"].update(yhat)

        if args.GP.is_gp:
            gp_training_params = {
                "feat": callback["neck_feat"].data,
                "loc": np.stack([lbl["lat"], lbl["lon"]], -1),
                "year": lbl["year"],
                "poi": lbl["poi_num"],
                "building": lbl["building_area"],
                "y": y.cpu(),
            }
            gp.append_training_params(**gp_training_params)

    r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
    rmse = math.sqrt(
        mean_squared_error(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
    )

    logging.info(f"[train] epoch {epoch}/{args.M.epochs} r2={r2:.3f} rmse={rmse:.4f}")

    writer.add_scalar("train/loss", meters["loss"].avg(), epoch)
    # writer.add_scalar("train/loss2", meters["loss2"].avg(), epoch)
    # writer.add_scalar("train/tri_loss", acc["train/tri_loss"], epoch)

    # writer.add_scalar("Train/r2", acc["train/r2"], epoch)
    # writer.add_scalar("Train/rmse", acc["train/rmse"], epoch)

    if args["FDS"]["is_fds"]:
        meters = {"y": Meter(), "img_fea": Meter(), "num_fea": Meter()}
        with torch.no_grad():
            for img, num, lbl, ind in loader:
                img = img.float().cuda()
                num = num.float().cuda()
                y = lbl["MPI3_fixed"].float().cuda()
                aux = {"epoch": epoch, "label": y}
                yhat, _, _ = model(img, num, aux)
                meters["img_fea"].update(callback["img_fea"].data)
                meters["num_fea"].update(callback["num_fea"].data)
                meters["y"].update(y)
        model.FDS1.update_last_epoch_stats(epoch)
        model.FDS1.update_running_stats(
            meters["img_fea"].cat(), meters["y"].cat(), epoch
        )
        model.FDS2.update_last_epoch_stats(epoch)
        model.FDS2.update_running_stats(
            meters["num_fea"].cat(), meters["y"].cat(), epoch
        )


def valid_epoch(args, model, callback, loader, writer, gp=None):
    global early_stop
    epoch = args.M.crt_epoch
    with torch.no_grad():
        model.eval()
        # criterion = HEMLoss(0)
        meters = {"loss": Meter(), "y": Meter(), "yhat": Meter()}
        for img, num, lbl, ind in loader:
            img = img.float().cuda()
            num = num.float().cuda()
            y = lbl["MPI3_fixed"].float().cuda()
            ind = ind.float().cuda()
            out = model(img, num)
            yhat, indhat, feat = out[0], out[1], out[2]
            if gp:
                gp.append_testing_params(
                    callback["neck_feat"].data,
                    np.stack([lbl["lat"], lbl["lon"]], -1),
                    lbl["year"],
                    lbl["poi_num"],
                    lbl["building_area"],
                )
            meters["y"].update(y)
            meters["yhat"].update(yhat)

        r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        rmse = math.sqrt(
            mean_squared_error(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        )

        logging.info(
            f"[valid] epoch {epoch}/{args.M.epochs} r2={r2:.3f} rmse={rmse:.4f} "
        )

        if gp:
            ygp = gp.gp_run(
                model.state_dict()["head1.0.weight"].cpu(),
                model.state_dict()["head1.0.bias"].cpu(),
            )
            r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
            rmse = math.sqrt(mean_squared_error(ygp, meters["y"].cat()).numpy().item())

            acc = {"valid/r2": r2, "valid/rmse": rmse}

            logging.info(
                f"[gp-valid] epoch {epoch}/{args.M.epochs} r2={r2:.3f} rmse={rmse:.4f} "
            )

        writer.add_scalar("valid/r2", r2, epoch)
        writer.add_scalar("valid/rmse", rmse, epoch)

    return r2


def test_epoch(args, model, callback, loader, writer, gp=None):
    with torch.no_grad():
        model.eval()
        epoch = args.M.crt_epoch
        model.load_state_dict(torch.load(args.M.best_weight_path))
        if gp:
            gp.restore(args.GP.best_gp_path)

        meters = {
            "y": Meter(),
            "yhat": Meter(),
            "name": Meter(),
            "lon": Meter(),
            "lat": Meter(),
            "ind": Meter(),
            "indhat": Meter(),
            "num": Meter(),
        }

        for img, num, lbl, ind in loader:
            img = img.float().cuda()
            num = num.float().cuda()
            y = lbl["MPI3_fixed"].float().cuda()
            ind = ind.float().cuda()
            out = model(img, num)
            yhat, indhat, feat = out[0], out[1], out[2]
            if args.GP.is_gp:
                gp.append_testing_params(
                    callback["neck_feat"].data,
                    np.stack([lbl["lat"], lbl["lon"]], -1),
                    lbl["year"],
                    lbl["poi_num"],
                    lbl["building_area"],
                )
            meters["y"].update(y)
            meters["yhat"].update(yhat)
            meters["name"].update(lbl["name"])
            meters["lon"].update(lbl["lon"])
            meters["lat"].update(lbl["lat"])
            meters["ind"].update(ind)
            meters["indhat"].update(indhat)
            meters["num"].update(num)

        r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        rmse = math.sqrt(
            mean_squared_error(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        )

        logging.info(f"[test] Testing with {args['M']['best_weight_path']}")
        logging.info(f"[test] r2={r2:.3f} rmse={rmse:.4f}")

        if gp:
            ygp = gp.gp_run(
                model.state_dict()["head1.0.weight"].cpu(),
                model.state_dict()["head1.0.bias"].cpu(),
            )
            r2 = r2_score(ygp, meters["y"].cat()).numpy().item()
            rmse = math.sqrt(mean_squared_error(ygp, meters["y"].cat()).numpy().item())
            logging.info(
                f"[gp-test] epoch {epoch}/{args.M.epochs} r2={r2:.3f} rmse={rmse:.4f} "
            )

        writer.add_scalar("test/r2", r2, epoch)
        writer.add_scalar("test/rmse", rmse, epoch)

        df = pd.DataFrame(
            {
                "name": meters["name"].cat(),
                "y": meters["y"].cat(),
                "yhat": meters["yhat"].cat(),
                "lon": meters["lon"].cat(),
                "lat": meters["lat"].cat(),
            }
        )
        df.to_csv(os.path.join(args.M.current_log_dir, "predict.csv"))
        torch.save(model, os.path.join(args.M.current_log_dir, "final_model.pt"))

        indicator_columns = [
            "road_length",
            "death_num",
            "building_area",
            "burnedCount",
            "conflict_num",
            "place_num",
            "poi_num",
            "water",
            "lat",
            "lon",
            "year",
        ]

        df1 = pd.DataFrame(
            meters["indhat"].cat().cpu().detach().numpy(), columns=indicator_columns
        )
        df2 = pd.DataFrame(
            meters["num"].cat().cpu().detach().numpy(), columns=indicator_columns
        )
        df1.to_csv(os.path.join(args.M.current_log_dir, "num_hat.csv"))
        df2.to_csv(os.path.join(args.M.current_log_dir, "num.csv"))


def pipline(args, train_list, valid_list, test_list):

    logging_setting(args.M.current_log_dir)
    writer = SummaryWriter(log_dir=args.M.current_log_dir)

    logging.info(
        f"[DataSize] train,validate,test: {len(train_list)},{len(valid_list)},{len(test_list)}"
    )

    # 4.定义 data loader
    LABELS = [label["MPI3_fixed"] for _, _, label, _ in mpi_dataset(args, train_list)]
    # WEIGHTS = get_lds_weights(LABELS)

    transform = A.Compose(
        [
            # A.RandomCrop(width=10, height=10),
            # A.ShiftScaleRotate(p=0.2),
            # A.HorizontalFlip(p=0.5),
            # A.Flip(p=0.1),
            # A.SafeRotate(p=0.2),
            # A.RandomBrightnessContrast(p=0.1),
        ]
    )
    train_loader = DataLoader(
        mpi_dataset(args, train_list),
        batch_size=args.M.batch_size,
        shuffle=True,
        num_workers=4,
        # sampler=torch.utils.data.WeightedRandomSampler(
        #     weights=WEIGHTS, num_samples=len(WEIGHTS), replacement=True
        # ),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        mpi_dataset(args, valid_list),
        batch_size=args.M.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        mpi_dataset(args, test_list),
        batch_size=args.M.batch_size,
        shuffle=False,
        num_workers=0,
    )
    # 5.获取model 并打印模型结构到文件
    train_model, callback = prepare_model(args)

    if args.M.best_weight_path is not None:
        train_model.load_state_dict(torch.load(args.M.best_weight_path))
        print("best weight loaded")

    # 6.初始化定义优化器、Scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, train_model.parameters()),
        lr=args["M"]["init_lr"],
    )
    scheduler = MultiStepLR(optimizer, **args.M.scheduler)

    # 7.开始训练，以最优的valid模型进行test
    early_stop = 0  # early stop
    gp = None
    start_epoch = args.M.crt_epoch
    end_epoch = args["M"]["epochs"] + 1
    for epoch in range(start_epoch, end_epoch):

        args.M.crt_epoch = epoch

        if args.GP.is_gp:
            gp = gp_model(
                sigma=args.GP.sigma,
                r_loc=args.GP.r_loc,
                r_year=args.GP.r_year,
                r_poi=args.GP.r_poi,
                r_building=args.GP.r_building,
                sigma_e=args.GP.sigma_e,
                sigma_b=args.GP.sigma_b,
            )
            gp.clear_params()
        # if args.GP.best_gp_path:
        #     gp.restore(args.GP.best_gp_path)

        # training
        if epoch % 1 == 0:
            train_epoch(
                args, train_model, callback, train_loader, optimizer, writer, gp
            )

        # validation
        if epoch > 10 and epoch % 5 == 0:
            acc = valid_epoch(args, train_model, callback, valid_loader, writer, gp)
            if acc > args.M.best_acc:
                args.M.best_acc = acc
                args.M.best_weight_path = os.path.join(
                    args.M.current_log_dir, "best_r2.pth"
                )
                torch.save(train_model.state_dict(), args.M.best_weight_path)
                if args.GP.is_gp:
                    args.GP.best_gp_path = os.path.join(
                        args.M.current_log_dir, "best_gp.pth"
                    )
                    gp.save(args.GP.best_gp_path)

                early_stop = 0
                # if best validation model, then testing
                test_model, test_callback = prepare_model(args)
                test_epoch(args, test_model, test_callback, test_loader, writer, gp)
            else:
                early_stop += 1
                logging.info(
                    f"Early Stop Counter {early_stop} of {args.M.max_early_stop}."
                )
            if early_stop >= args.M.max_early_stop:
                break

        scheduler.step()
    """
    final test
    """
    test_model, callback = prepare_model(args)
    test_epoch(args, test_model, callback, test_loader, writer, gp)

    # save args
    save_yaml(args, os.path.join(args.M.current_log_dir, "config.yaml"))
    return "OK"


def train(cfg_path="Config/swint.yaml", tag="base", index=None, restore=False):
    args = parse_yaml(cfg_path)
    args.M.parent_log_dir = os.path.join(args.M.root_log_dir, tag)
    setup_seed(args.M.seed)
    prepare_datalist(args)
    processes = []
    if args.M.split_method == "cv" and index == None:
        run_indexs = range(1, args.M.k_fold + 1)
    elif args.M.split_method == "cv" and index != None:
        run_indexs = [index]
    elif args.M.split_method == "holdout":
        run_indexs = [1]
    else:
        raise NotImplementedError
    for i in run_indexs:
        # args.M.current_log_dir = os.path.join(args.M.current_log_dir, tag, str(index))
        data = parse_yaml(os.path.join(args.M.parent_log_dir, str(i) + ".yaml"))
        train_list = [os.path.join(args.D.data_dir, i) for i in data.train_list]
        valid_list = [os.path.join(args.D.data_dir, i) for i in data.valid_list]
        test_list = [os.path.join(args.D.data_dir, i) for i in data.test_list]

        args.M.current_log_dir = os.path.join(args.M.parent_log_dir, str(i))

        # 默认情况下，会清空已有的current_log_dir
        if os.path.exists(args.M.current_log_dir):
            if restore:
                args = parse_yaml(os.path.join(args.M.current_log_dir, "config.yaml"))
            else:
                shutil.rmtree(args.M.current_log_dir)
                os.mkdir(args.M.current_log_dir)

        os.environ["CUDA_VISIBLE_DEVICES"] = prepare_gpu()
        print(f"[INFO] start the pipline: {i}.yaml")
        p = mp.Process(target=pipline, args=(args, train_list, valid_list, test_list))
        p.start()
        processes.append(p)
        time.sleep(10)
    for p in processes:
        p.join()


def stat(logname):
    items = {"name": [], "seed": [], "r2": [], "rmse": [], "mape": []}
    log_list = glob.glob(f"{logname}/*/")
    for log in log_list:
        cfg = parse_yaml(os.path.join(log, "config.yaml"))
        r2, rmse, mape = None, None, None
        with open(os.path.join(log, "run.log"), "r") as f:
            lines = f.readlines()
            line = lines[-1].strip()
            pattern = re.compile(r"(?<=r2.)\d+\.?\d*")
            r2 = pattern.findall(line)[0]
            pattern = re.compile(r"(?<=mse.)\d+\.?\d*")
            rmse = pattern.findall(line)[0]
            mape = 0
        items["name"].append(log)
        items["seed"].append(cfg["M"]["seed"])
        items["r2"].append(r2)
        items["rmse"].append(rmse)
        items["mape"].append(mape)
    items["name"].append("Average")
    items["seed"].append("-1")
    items["r2"].append(round(np.average([float(i) for i in items["r2"]]), 4))
    items["rmse"].append(round(np.average([float(i) for i in items["rmse"]]), 4))
    items["mape"].append(round(np.average([float(i) for i in items["mape"]]), 3))
    # print(items)
    df = pd.DataFrame(items, index=None)
    df.to_csv(f"{logname}/STAT.csv", index=False)
    print(df)


def predict(log_dir):
    # 删除以前生成的日志文件
    os.system(f"rm {log_dir}/events*")
    os.environ["CUDA_VISIBLE_DEVICES"] = prepare_gpu()
    writer = SummaryWriter(log_dir=log_dir)

    meters = {
        "y": Meter(),
        "yhat": Meter(),
        "name": Meter(),
        "lon": Meter(),
        "lat": Meter(),
        "ind": Meter(),
        "indhat": Meter(),
        "img_fea": Meter(),
        "num_fea": Meter(),
        "neck_feat": Meter(),
    }
    for fold in glob.glob(f"{log_dir}/*/"):
        cv_index = os.path.basename(os.path.dirname(fold))
        args = parse_yaml(os.path.join(fold, "config.yaml"))
        setup_seed(args.M.seed)
        data_list = parse_yaml(os.path.join(f"{log_dir}/{cv_index}.yaml"))
        test_list = data_list["test_list"]
        test_list = [os.path.join(args["D"]["data_dir"], i) for i in test_list]
        model, callback = prepare_model(args)
        loader = DataLoader(
            mpi_dataset(args, test_list),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            model.eval()
            model.load_state_dict(torch.load(args["M"]["best_weight_path"]))
            if args["GP"]["is_gp"]:
                gp = gp_model(sigma=1, r_loc=2.5, r_year=10, sigma_e=0.32, sigma_b=0.01)
                gp.restore(args["GP"]["best_gp_path"])

            for img, num, lbl, ind in loader:
                img = img.float().cuda()
                num = num.float().cuda()
                y = lbl["MPI3_fixed"].float().cuda()
                ind = ind.float().cuda()
                yhat, indhat, _ = model(img, num)
                # if args["GP"]["is_gp"]:
                #     gp.append_testing_params(
                #         callback["neck_feat"].data,
                #         np.stack([lbl["lat"], lbl["lon"]], -1),
                #         lbl["year"].item(),
                #         lbl["poi_num"].item(),
                #         lbl["poi_num"].item(),
                #     )
                meters["y"].update(y)
                meters["yhat"].update(yhat)
                meters["name"].update(lbl["name"])
                meters["lon"].update(lbl["lon"])
                meters["lat"].update(lbl["lat"])
                meters["ind"].update(ind)
                meters["indhat"].update(indhat)
                meters["img_fea"].update(callback["img_fea"].data)
                meters["num_fea"].update(callback["num_fea"].data)
                meters["neck_feat"].update(callback["neck_feat"].data)

            # if args["GP"]["is_gp"]:
            #     ygp = gp.gp_run(
            #         model.state_dict()["head1.0.weight"].cpu(),
            #         model.state_dict()["head1.0.bias"].cpu(),
            #     )

            # r2 = r2_score(ygp, meters["y"].cat()).numpy().item()
            # rmse = math.sqrt(mean_squared_error(ygp, meters["y"].cat()).numpy().item())

            # writer.add_scalar("Test/r2", acc["test/r2"], epoch)
            # writer.add_scalar("Test/rmse", acc["test/rmse"], epoch)
            # writer.add_histogram("Test/img_fea", callback["img_fea"].data, epoch)
            # writer.add_histogram("Test/num_fea", callback["num_fea"].data, epoch)

        df = {
            "name": meters["name"].cat(),
            "y": meters["y"].cat(),
            "yhat": meters["yhat"].cat(),
            "lon": meters["lon"].cat(),
            "lat": meters["lat"].cat(),
            "img_fea": meters["img_fea"].cat(),
            "num_fea": meters["num_fea"].cat(),
            "neck_feat": meters["neck_feat"].cat(),
        }

        count_analysis(df, cv_index, writer)
        feature_statistics(df, cv_index, writer)
        chart_1_fold_result(df, cv_index, writer)

    writer.close()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")  # good solution !!!!
    # fire.Fire(run_1_fold)
    fire.Fire()
