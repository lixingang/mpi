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

# from multiprocessing import Process
import torch.multiprocessing as mp

# import in-project packages
from Losses.loss import *
from Losses.tri_loss import calc_triplet_loss
from Models import *
from Models.LDS import get_lds_weights
from Datasets.mpi_datasets import mpi_dataset
from Utils.base import parse_yaml, parse_log
from Utils.base import setup_seed, Meter, SaveOutput, split_train_valid


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


def get_model(args):
    assert args["M"]["model"].lower() in {"vit", "mlp", "swint"}
    model = None
    callback = {}
    callback["last_fea"] = SaveOutput()
    # callback["weights_fea"] = SaveOutput()

    if args["M"]["model"].lower() == "mlp":
        model = MLP(
            len(args["D"]["img_keys"]) * args["D"]["in_channel"],
            len(args["D"]["num_keys"]),
        ).cuda()
        print(model, file=open(os.path.join(args["M"]["log_dir"], "model.txt"), "a"))
        hook1 = model.fclayer[2].register_forward_hook(callback["last_fea"])
        hook2 = model.fclayer[4].register_forward_hook(callback["weights_fea"])

    elif args["M"]["model"].lower() == "swint":
        callback["img_fea"] = SaveOutput()
        callback["num_fea"] = SaveOutput()
        model = SwinTransformer(
            img_size=args["M"]["img_size"],
            patch_size=4,
            in_chans=(len(args["D"]["img_keys"]), len(args["D"]["num_keys"])),
            num_classes=10,
            embed_dim=54,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=int(args["M"]["img_size"] / 32),
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.00,
            drop_path_rate=0.0,
        ).cuda()

        model_parameter = summary(
            model,
            input_size=[
                (
                    1,
                    len(args["D"]["img_keys"]),
                    args["M"]["img_size"],
                    args["M"]["img_size"],
                ),
                (1, len(args["D"]["num_keys"])),
            ],
            verbose=0,
        )
        print(
            model_parameter,
            file=open(os.path.join(args["M"]["log_dir"], "model_parameter.txt"), "w"),
        )
        print(
            model,
            file=open(os.path.join(args["M"]["log_dir"], "module_name.txt"), "w"),
        )

        hook1 = model.head1[0].register_forward_hook(callback["last_fea"])
        # hook2 = model.head.register_forward_hook(callback["weights_fea"])
        hook3 = model.avgpool.register_forward_hook(callback["img_fea"])
        # hook4 = model.num_layers[-1].register_forward_hook(callback["num_fea"])

    return model, callback


def train_epoch(args, model, callback, loader, optimizer, writer, gp=None):

    model.train()
    epoch = args["M"]["crt_epoch"]
    meters = {
        "loss": Meter(),
        "y": Meter(),
        "yhat": Meter(),
        "loss1": Meter(),
        "loss2": Meter(),
        "tri_loss": Meter(),
    }
    # criterion = BMCLoss(init_noise_sigma=0.1)
    # optimizer.add_param_group({"params": criterion.noise_sigma, "name": "noise_sigma"})
    for img, num, lbl, ind in loader:
        img = img.float().cuda()
        num = num.float().cuda()
        y = lbl["MPI3_fixed"].float().cuda()
        ind = ind.float().cuda()

        # aux = {"epoch": epoch, "label": y} if args["FDS"]["is_fds"] else {}
        # aux = {}
        out = model(img, num)
        yhat, indhat, feat = out[0], out[1], out[2]
        loss1 = weighted_mse_loss(yhat, y)
        # loss1 = weighted_huber_loss(y, yhat,get_lds_weights(y))
        # loss1 = weighted_huber_loss(y, yhat, get_lds_weights(y), 1)
        # loss1 = weighted_huber_loss(yhat, y, get_lds_weights(y))
        loss2 = weighted_mse_loss(indhat, ind)
        # loss3 = weighted_huber_loss(ind, indhat)
        # print(loss1, loss3)
        tri_loss = calc_triplet_loss(feat, y - 0, 0.8 - 0)

        # loss = loss1 + 0.001 * loss2 + 0.001 * tri_loss
        loss = loss1 + 0.001 * loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        meters["loss"].update(loss1)
        # meters["loss2"].update(loss2)
        # meters["tri_loss"].update(tri_loss)
        meters["y"].update(y)
        meters["yhat"].update(yhat)

        if args["GP"]["is_gp"]:
            gp_training_params = {
                "feat": callback["last_fea"].data,
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
    acc = {
        "train/loss": meters["loss"].avg(),
        # "train/loss2": meters["loss2"].avg(),
        # "train/tri_loss": meters["tri_loss"].avg(),
        # "train/r2": r2,
        # "train/rmse": rmse,
    }

    logging.info(
        f"[train] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} rmse={rmse:.4f}"
    )

    writer.add_scalar("train/loss", acc["train/loss"], epoch)
    # writer.add_scalar("train/loss2", acc["train/loss2"], epoch)
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
                # meters["num_fea"].update(callback["num_fea"].data)
                meters["y"].update(y)
        model.FDS1.update_last_epoch_stats(epoch)
        model.FDS1.update_running_stats(
            meters["img_fea"].cat(), meters["y"].cat(), epoch
        )

    return acc


def valid_epoch(args, model, callback, loader, writer, gp=None):
    global early_stop
    epoch = args["M"]["crt_epoch"]
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
                    callback["last_fea"].data,
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
        acc = {"valid/r2": r2, "valid/rmse": rmse}

        logging.info(
            f"[valid] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} rmse={rmse:.4f} "
        )

        if gp:
            ygp = gp.gp_run(
                model.state_dict()["head1.1.weight"].cpu(),
                model.state_dict()["head1.1.bias"].cpu(),
            )
            r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
            rmse = math.sqrt(mean_squared_error(ygp, meters["y"].cat()).numpy().item())

            acc = {"valid/r2": r2, "valid/rmse": rmse}

            logging.info(
                f"[gp-valid] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} rmse={rmse:.4f} "
            )

        writer.add_scalar("valid/r2", acc["valid/r2"], epoch)
        writer.add_scalar("valid/rmse", acc["valid/rmse"], epoch)

    return acc


def test_epoch(args, model, callback, loader, writer, gp=None):
    with torch.no_grad():
        model.eval()
        epoch = args["M"]["crt_epoch"]
        model.load_state_dict(torch.load(args["M"]["best_weight_path"]))
        if gp:
            gp.restore(args["GP"]["best_gp_path"])

        meters = {
            "y": Meter(),
            "yhat": Meter(),
            "name": Meter(),
            "lon": Meter(),
            "lat": Meter(),
            "ind": Meter(),
            "indhat": Meter(),
        }
        for img, num, lbl, ind in loader:
            img = img.float().cuda()
            num = num.float().cuda()
            y = lbl["MPI3_fixed"].float().cuda()
            ind = ind.float().cuda()
            out = model(img, num)
            yhat, indhat, feat = out[0], out[1], out[2]
            if args["GP"]["is_gp"]:
                gp.append_testing_params(
                    callback["last_fea"].data,
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

        r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        rmse = math.sqrt(
            mean_squared_error(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        )

        acc = {"test/r2": r2, "test/rmse": rmse}

        logging.info(f"[test] Testing with {args['M']['best_weight_path']}")
        logging.info(f"[test] r2={r2:.3f} rmse={rmse:.4f}")

        if gp:
            ygp = gp.gp_run(
                model.state_dict()["head1.1.weight"].cpu(),
                model.state_dict()["head1.1.bias"].cpu(),
            )
            r2 = r2_score(ygp, meters["y"].cat()).numpy().item()
            rmse = math.sqrt(mean_squared_error(ygp, meters["y"].cat()).numpy().item())
            acc = {"test/r2": r2, "test/rmse": rmse}

            logging.info(
                f"[gp-test] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} rmse={rmse:.4f} "
            )

        writer.add_scalar("test/r2", acc["test/r2"], epoch)
        writer.add_scalar("test/rmse", acc["test/rmse"], epoch)
        # writer.add_histogram("Test/img_fea", callback["img_fea"].data, epoch)
        # writer.add_histogram("Test/num_fea", callback["num_fea"].data, epoch)

        df = pd.DataFrame(
            {
                "name": meters["name"].cat(),
                "y": meters["y"].cat(),
                "yhat": meters["yhat"].cat(),
                "lon": meters["lon"].cat(),
                "lat": meters["lat"].cat(),
            }
        )
        df.to_csv(os.path.join(args["M"]["log_dir"], "predict.csv"))
        indicator_columns = [
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
        # df1 = pd.DataFrame(
        #     meters["indhat"].cat().cpu().detach().numpy(), columns=indicator_columns
        # )
        # df2 = pd.DataFrame(
        #     meters["ind"].cat().cpu().detach().numpy(), columns=indicator_columns
        # )
        # df1.to_csv(os.path.join(args["M"]["log_dir"], "ind_hat.csv"))
        # df2.to_csv(os.path.join(args["M"]["log_dir"], "ind.csv"))
        return acc


def pipline(args, train_list, valid_list, test_list):

    logging_setting(args["M"]["log_dir"])
    writer = SummaryWriter(log_dir=args["M"]["log_dir"])

    logging.info(
        f"[DataSize] train,validate,test: {len(train_list)},{len(valid_list)},{len(test_list)}"
    )

    # 4.定义 data loader
    LABELS = [label["MPI3_fixed"] for _, _, label, _ in mpi_dataset(args, train_list)]
    WEIGHTS = get_lds_weights(LABELS)

    train_loader = DataLoader(
        mpi_dataset(args, train_list),
        batch_size=args["M"]["batch_size"],
        shuffle=True,
        num_workers=3,
        # sampler=torch.utils.data.WeightedRandomSampler(
        #     weights=WEIGHTS, num_samples=len(WEIGHTS), replacement=True
        # ),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        mpi_dataset(args, valid_list),
        batch_size=args["M"]["batch_size"],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        mpi_dataset(args, test_list),
        batch_size=args["M"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    # 5.获取model 并打印模型结构到文件
    train_model, callback = get_model(args)

    if args["M"]["restore_weight"] is not None:
        train_model.load_state_dict(torch.load(args["M"]["restore_weight"]))
        print("weight loaded")

    # 6.初始化定义优化器、Scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, train_model.parameters()),
        lr=args["M"]["init_lr"],
    )
    scheduler = MultiStepLR(optimizer, **args["M"]["scheduler"])

    # 7.开始训练，以最优的valid模型进行test
    early_stop = 0  # early stop
    gp = None
    start_epoch = args["M"]["crt_epoch"]
    end_epoch = args["M"]["epochs"] + 1
    for epoch in range(start_epoch, end_epoch):

        args["M"]["crt_epoch"] = epoch

        if args["GP"]["is_gp"]:
            gp = gp_model(
                sigma=args["GP"]["sigma"],
                r_loc=args["GP"]["r_loc"],
                r_year=args["GP"]["r_year"],
                r_poi=args["GP"]["r_poi"],
                r_building=args["GP"]["r_building"],
                sigma_e=args["GP"]["sigma_e"],
                sigma_b=args["GP"]["sigma_b"],
            )
            gp.clear_params()
        # if args["GP"]["best_gp_path"]:
        #     gp.restore(args["GP"]["best_gp_path"])

        # training
        if epoch % 1 == 0:
            _ = train_epoch(
                args, train_model, callback, train_loader, optimizer, writer, gp
            )

        # validation
        if epoch % 5 == 0:
            valid_acc = valid_epoch(
                args, train_model, callback, valid_loader, writer, gp
            )
            if valid_acc["valid/r2"] > args["M"]["best_acc"]:
                args["M"]["best_acc"] = valid_acc["valid/r2"]
                args["M"]["best_weight_path"] = os.path.join(
                    args["M"]["log_dir"], "best_r2.pth"
                )
                torch.save(train_model.state_dict(), args["M"]["best_weight_path"])
                if args["GP"]["is_gp"]:
                    args["GP"]["best_gp_path"] = os.path.join(
                        args["M"]["log_dir"], "best_gp.pth"
                    )
                    gp.save(args["GP"]["best_gp_path"])

                early_stop = 0
                # if best validation model, then testing

            else:
                early_stop += 1
                logging.info(
                    f"Early Stop Counter {early_stop} of {args['M']['max_early_stop']}."
                )

            test_model, test_callback = get_model(args)
            _ = test_epoch(args, test_model, test_callback, test_loader, writer, gp)

            if early_stop >= args["M"]["max_early_stop"]:
                break

        scheduler.step()

        # if epoch % 20 == 0:
        #     logging.info(
        #         f"epoch{epoch}, Current learning rate: {scheduler.get_last_lr()}"
        #     )

    """
    final test
    """
    test_model, callback = get_model(args)
    _ = test_epoch(args, test_model, callback, test_loader, writer, gp)

    # save args
    os.makedirs(args["M"]["log_dir"], exist_ok=True)
    with open(os.path.join(args["M"]["log_dir"], "config.yaml"), "w") as f:
        yaml.dump(args, f)
    return "OK"


def prepare_fold(cfg_path="origin.yaml", tag="lds"):

    args = parse_yaml(cfg_path)
    setup_seed(args["M"]["seed"])

    log_dir = os.path.join(
        args["M"]["log_dir"],
        os.path.splitext(os.path.basename(cfg_path))[0] + "_" + tag,
    )

    if os.path.exists(log_dir):
        message = input(
            "[INFO] found duplicate directories, whether to overwrite (Y/N/R) "
        )
        if message.lower() == "y":
            print("[INFO] deleting the existing logs...")
            shutil.rmtree(log_dir)
        elif message.lower() == "r":
            print("[INFO] restoring the existing logs...")
            return "restore"
        else:
            print("[INFO] canceled")
            sys.exit(0)

    os.makedirs(log_dir, exist_ok=True)

    data_list = pd.read_csv(args["D"]["source"])["name"].to_numpy()
    value_list = pd.read_csv(args["D"]["source"])["MPI3_fixed"].to_numpy()
    sorted_id = sorted(
        range(len(value_list)), key=lambda k: value_list[k], reverse=False
    )
    data_list = data_list[sorted_id]
    value_list = value_list[sorted_id]
    # kf = KFold(n_splits=5, shuffle=True, random_state=args["M"]["seed"])

    fold = {"name": [], "value": []}
    k = args["M"]["k"]
    for i in range(k):
        fold["name"].append([v for v in range(0 + i, len(data_list), k)])
        fold["value"].append([v for v in range(0 + i, len(value_list), k)])
    # for i, (train_part, test_index) in enumerate(kf.split(data_list)):
    #     train_index, valid_index = split_train_valid(train_part, [0.8, 0.2])
    for i in range(k):
        test_part = fold["name"][i]
        train_part = {"name": [], "value": []}
        for j in range(k):
            if j == i:
                continue
            else:
                train_part["name"] = train_part["name"] + fold["name"][j]
                train_part["value"] = train_part["value"] + fold["value"][j]
        train_index, valid_index = split_train_valid(
            np.array(train_part["name"]), [0.75, 0.25]
        )
        test_index = np.array(test_part)

        yamlname = (
            os.path.join(
                args["M"]["log_dir"],
                os.path.splitext(os.path.basename(cfg_path))[0] + "_" + tag,
                str(i + 1),
            )
            + ".yaml"
        )
        with open(yamlname, "w") as f:
            yaml.dump(
                {
                    "log_dir": os.path.join(
                        args["M"]["log_dir"],
                        args["M"]["model"] + "_" + tag,
                    ),
                    "train_list": data_list[train_index].tolist(),
                    "valid_list": data_list[valid_index].tolist(),
                    "test_list": data_list[test_index].tolist(),
                    "train_value": value_list[train_index].tolist(),
                    "valid_value": value_list[valid_index].tolist(),
                    "test_value": value_list[test_index].tolist(),
                },
                f,
            )
        print(f"[INFO] creating C-V record: {yamlname}")
        # time.sleep(1)

    return log_dir


def run_restore(source_log_dir="Logs/swint_test"):
    processes = []
    for index in [1, 2, 3, 5]:

        # select gpu
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >.tmp")
        memory_gpu = [int(x.split()[2]) for x in open(".tmp", "r").readlines()]
        while max(memory_gpu) < 5500:
            time.sleep(20)
            print("[INFO] Query the idle GPU ...")
            os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >.tmp")
            memory_gpu = [int(x.split()[2]) for x in open(".tmp", "r").readlines()]

        print(f"[INFO] start the pipline: {str(index)}.yaml")
        print(f"[INFO] GPU:{np.argmax(memory_gpu)} was selected")
        best_gpu = int(np.argmax(memory_gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

        if not os.path.isdir(os.path.join(source_log_dir, str(index))):
            continue
        args = parse_yaml(os.path.join(source_log_dir, str(index), "config.yaml"))
        setup_seed(args["M"]["seed"])

        print(f"[INFO] loading C-V record: {str(index)}.yaml")
        data = parse_yaml(os.path.join(source_log_dir, str(index) + ".yaml"))
        train_list = data["train_list"]
        valid_list = data["valid_list"]
        test_list = data["test_list"]
        train_list = [os.path.join(args["D"]["data_dir"], i) for i in train_list]
        valid_list = [os.path.join(args["D"]["data_dir"], i) for i in valid_list]
        test_list = [os.path.join(args["D"]["data_dir"], i) for i in test_list]

        # pipline(
        #     args,
        #     train_list,
        #     valid_list,
        #     test_list,
        # )

        p = mp.Process(target=pipline, args=(args, train_list, valid_list, test_list))
        p.start()
        processes.append(p)
        time.sleep(10)
    for p in processes:
        p.join()


def run_1_fold(cfg_path="origin.yaml", tag="base", index=1):
    # select gpu
    #  Query the idle GPU ... , generate .tmp file in current folder
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >.tmp")
    memory_gpu = [int(x.split()[2]) for x in open(".tmp", "r").readlines()]
    while max(memory_gpu) < 5500:
        time.sleep(20)
        print("[INFO] Query the idle GPU ...")
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >.tmp")
        memory_gpu = [int(x.split()[2]) for x in open(".tmp", "r").readlines()]

    print(f"[INFO] start the pipline: {index}.yaml")
    print(f"[INFO] GPU:{np.argmax(memory_gpu)} was selected")
    best_gpu = int(np.argmax(memory_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

    # start pipline
    log_dir = prepare_fold(cfg_path, tag)
    args = parse_yaml(cfg_path)
    setup_seed(args["M"]["seed"])

    assert os.path.exists(log_dir), "目录不存在, 请先运行prepare_fold命令."
    args["M"]["log_dir"] = os.path.join(log_dir, str(index))
    print(f"[INFO] loading C-V record: {index}.yaml")
    data = parse_yaml(os.path.join(log_dir, str(index) + ".yaml"))
    train_list = data["train_list"]
    valid_list = data["valid_list"]
    test_list = data["test_list"]
    train_list = [os.path.join(args["D"]["data_dir"], i) for i in train_list]
    valid_list = [os.path.join(args["D"]["data_dir"], i) for i in valid_list]
    test_list = [os.path.join(args["D"]["data_dir"], i) for i in test_list]
    print(f"[INFO] start the pipline: {index}.yaml")
    pipline(
        args,
        train_list,
        valid_list,
        test_list,
    )


def run_all(cfg_path="Config/swint.yaml", tag="base", indexs=None):
    log_dir = prepare_fold(cfg_path, tag)
    args = parse_yaml(cfg_path)
    setup_seed(args["M"]["seed"])

    assert os.path.exists(log_dir), "目录不存在, 请先运行prepare_fold命令."

    processes = []
    if indexs == None:
        run_indexs = range(1, args["M"]["k"] + 1)
    else:
        run_indexs = indexs
    for index in run_indexs:
        args["M"]["log_dir"] = os.path.join(log_dir, str(index))
        print(f"[INFO] loading C-V record {index}.yaml")
        data = parse_yaml(os.path.join(log_dir, str(index) + ".yaml"))
        train_list = data["train_list"]
        valid_list = data["valid_list"]
        test_list = data["test_list"]
        train_list = [os.path.join(args["D"]["data_dir"], i) for i in train_list]
        valid_list = [os.path.join(args["D"]["data_dir"], i) for i in valid_list]
        test_list = [os.path.join(args["D"]["data_dir"], i) for i in test_list]

        # pipline(args, train_list, valid_list, test_list)
        print("[INFO] Query the idle GPU ...")
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >.tmp")
        memory_gpu = [int(x.split()[2]) for x in open(".tmp", "r").readlines()]
        while max(memory_gpu) < 5500:
            time.sleep(20)
            print("[INFO] Query the idle GPU ...")
            os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >.tmp")
            memory_gpu = [int(x.split()[2]) for x in open(".tmp", "r").readlines()]

        print(f"[INFO] start the pipline: {index}.yaml")
        print(f"[INFO] GPU:{np.argmax(memory_gpu)} was selected")
        best_gpu = int(np.argmax(memory_gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        # os.system("rm tmp")  # 删除临时生成的 tmp 文件
        p = mp.Process(target=pipline, args=(args, train_list, valid_list, test_list))
        p.start()
        processes.append(p)
        time.sleep(10)
    for p in processes:
        p.join()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")  # good solution !!!!
    # fire.Fire(run_1_fold)
    fire.Fire()
