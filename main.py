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
from multiprocessing import Process

# import in-project packages
from Losses.loss import *
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
    callback["weights_fea"] = SaveOutput()
    if args["M"]["model"].lower() == "vit":
        model = ViT(
            num_patches=len(args["D"]["img_keys"]) + len(args["D"]["num_keys"]),
            patch_dim=args["D"]["in_channel"],
            num_classes=10,
            dim=args["VIT"]["vit_dim"],
            depth=args["VIT"]["vit_depth"],
            heads=args["VIT"]["vit_heads"],
            mlp_dim=args["VIT"]["vit_mlp_dim"],
            dropout=args["VIT"]["vit_dropout"],
            emb_dropout=args["VIT"]["vit_dropout"],
        ).cuda()
        print(model, file=open(os.path.join(args["M"]["log_dir"], "model.txt"), "a"))
        hook1 = model.mlp_head[0].register_forward_hook(callback["last_fea"])
        hook2 = model.mlp_head[1].register_forward_hook(callback["weights_fea"])

    elif args["M"]["model"].lower() == "mlp":
        model = MLP(
            len(args["D"]["img_keys"]) * args["D"]["in_channel"],
            len(args["D"]["num_keys"]),
        ).cuda()
        print(model, file=open(os.path.join(args["M"]["log_dir"], "model.txt"), "a"))
        hook1 = model.fclayer[2].register_forward_hook(callback["last_fea"])
        hook2 = model.fclayer[4].register_forward_hook(callback["weights_fea"])

    elif args["M"]["model"].lower() == "swint":
        model = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=(len(args["D"]["img_keys"]), len(args["D"]["num_keys"])),
            num_classes=10,
            embed_dim=54,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.000,
        ).cuda()

        model_parameter = summary(
            model,
            input_size=[
                (1, len(args["D"]["img_keys"]), 224, 224),
                (1, len(args["D"]["num_keys"])),
            ],
            verbose=0,
        )
        print(
            model_parameter,
            file=open(os.path.join(args["M"]["log_dir"], "model.txt"), "w"),
        )

        hook1 = model.avgpool.register_forward_hook(callback["last_fea"])
        hook2 = model.head.register_forward_hook(callback["weights_fea"])

    return model, callback


def train_epoch(args, model, callback, loader, optimizer, writer, gp=None):

    model.train()
    epoch = args["M"]["crt_epoch"]
    meters = {"loss": Meter(), "y": Meter(), "yhat": Meter()}
    for img, num, lbl, ind in loader:
        img = img.float().cuda()
        num = num.float().cuda()
        y = lbl["MPI3_fixed"].float().cuda()
        ind = ind.float().cuda()

        aux = {"epoch": epoch, "label": y} if args["FDS"]["is_fds"] else {}
        # aux = {}
        yhat, indhat = model(img, num, aux)
        # loss1 = torch.nn.functional.mse_loss(yhat, y)
        loss1 = weighted_huber_loss(yhat, y, get_lds_weights(y))
        loss2 = torch.nn.functional.mse_loss(ind, indhat)

        loss = loss1 + 0.5 * loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        meters["loss"].update(loss)
        meters["y"].update(y)
        meters["yhat"].update(yhat)

        if args["GP"]["is_gp"]:
            gp_training_params = {
                "feat": callback["last_fea"].data,
                "year": lbl["year"],
                "loc": np.stack([lbl["lat"], lbl["lon"]], -1),
                "y": y.cpu(),
            }
            gp.append_training_params(**gp_training_params)

    r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
    rmse = math.sqrt(
        mean_squared_error(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
    )
    acc = {"train/loss": meters["loss"].avg(), "train/r2": r2, "train/rmse": rmse}

    logging.info(
        f"[train] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} rmse={rmse:.4f}"
    )

    writer.add_scalar("Train/loss", acc["train/loss"], epoch)
    writer.add_scalar("Train/r2", acc["train/r2"], epoch)
    writer.add_scalar("Train/rmse", acc["train/rmse"], epoch)

    if args["FDS"]["is_fds"]:
        meters = {"y": Meter(), "feat": Meter()}
        with torch.no_grad():
            for img, num, lbl, ind in loader:
                img = img.cuda()
                num = num.cuda()
                y = lbl["MPI3_fixed"].cuda()
                other = {"epoch": epoch, "label": y}
                yhat, _ = model(img, num, other)
                meters["feat"].update(callback["last_fea"].data)
                meters["y"].update(y)
        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(meters["feat"].cat(), meters["y"].cat(), epoch)
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
            yhat, indhat = model(img, num)
            loss1 = weighted_huber_loss(yhat, y, get_lds_weights(y))
            loss2 = torch.nn.functional.mse_loss(indhat, ind)
            loss = loss1 + loss2
            meters["loss"].update(loss)
            if gp:
                gp.append_testing_params(
                    callback["last_fea"].data,
                    lbl["year"],
                    np.stack([lbl["lat"], lbl["lon"]], -1),
                )
            meters["y"].update(y)
            meters["yhat"].update(yhat)

        if gp:
            print(model.state_dict().keys())
            indhat = gp.gp_run(
                model.state_dict()["fclayer.3.weight"].cpu(),
                model.state_dict()["fclayer.3.bias"].cpu(),
            ).cuda()

        r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        rmse = math.sqrt(
            mean_squared_error(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        )
        acc = {"valid/loss": meters["loss"].avg(), "valid/r2": r2, "valid/rmse": rmse}

        logging.info(
            f"[valid] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} rmse={rmse:.4f} "
        )

        writer.add_scalar("Valid/loss", acc["valid/loss"], epoch)
        writer.add_scalar("Valid/r2", acc["valid/r2"], epoch)
        writer.add_scalar("Valid/rmse", acc["valid/rmse"], epoch)

    return acc


def test_epoch(args, model, callback, loader, writer, gp=None):
    with torch.no_grad():
        model.eval()
        epoch = args["M"]["crt_epoch"]
        model.load_state_dict(torch.load(args["M"]["best_weight_path"]))
        if gp:
            gp.restore(args["M"]["best_gp_path"])

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
            yhat, indhat = model(img, num)
            if args["GP"]["is_gp"]:
                gp.append_testing_params(
                    callback["last_fea"].data,
                    lbl["year"],
                    np.stack([lbl["lat"], lbl["lon"]], -1),
                )
            meters["y"].update(y)
            meters["yhat"].update(yhat)
            meters["name"].update(lbl["name"])
            meters["lon"].update(lbl["lon"])
            meters["lat"].update(lbl["lat"])
            meters["ind"].update(ind)
            meters["indhat"].update(indhat)

        np.savetxt(
            os.path.join(args["M"]["log_dir"], "weight_features.csv"),
            meters["indhat"].cat().cpu().detach().numpy(),
            delimiter=",",
            fmt="%.3f",
        )
        np.savetxt(
            os.path.join(args["M"]["log_dir"], "weight_indicator.csv"),
            meters["ind"].cat().cpu().detach().numpy(),
            delimiter=",",
            fmt="%.3f",
        )
        if args["GP"]["is_gp"]:
            yhat = gp.gp_run(
                model.state_dict()["fclayer.3.weight"].cpu(),
                model.state_dict()["fclayer.3.bias"].cpu(),
            ).cuda()

        r2 = r2_score(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        rmse = math.sqrt(
            mean_squared_error(meters["yhat"].cat(), meters["y"].cat()).numpy().item()
        )
        acc = {"test/r2": r2, "test/rmse": rmse}

        logging.info(f"[test] Testing with {args['M']['best_weight_path']}")
        logging.info(f"[test] r2={r2:.3f} rmse={rmse:.4f}")

        writer.add_scalar("Test/r2", acc["test/r2"], epoch)
        writer.add_scalar("Test/rmse", acc["test/rmse"], epoch)

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

        return acc


def pipline(args, train_list, valid_list, test_list):
    if os.path.exists(args["M"]["log_dir"]):
        shutil.rmtree(args["M"]["log_dir"])
    logging_setting(args["M"]["log_dir"])
    writer = SummaryWriter(log_dir=args["M"]["log_dir"])

    logging.info(
        f"[DataSize] train,validate,test: {len(train_list)},{len(valid_list)},{len(test_list)}"
    )

    # 4.定义 data loader
    train_loader = DataLoader(
        mpi_dataset(args, train_list),
        batch_size=args["M"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        mpi_dataset(args, valid_list),
        batch_size=args["M"]["batch_size"],
        shuffle=True,
        num_workers=4,
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
            gp = gp_model(sigma=1, r_loc=2.5, r_year=3, sigma_e=0.32, sigma_b=0.01)
            gp.clear_params()

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
            if valid_acc["valid/rmse"] < args["M"]["best_acc"]:
                args["M"]["best_acc"] = valid_acc["valid/rmse"]
                args["M"]["best_weight_path"] = os.path.join(
                    args["M"]["log_dir"], f"ep{epoch}.pth"
                )
                torch.save(train_model.state_dict(), args["M"]["best_weight_path"])
                if args["GP"]["is_gp"]:
                    args["M"]["best_gp_path"] = args["M"]["best_weight_path"].replace(
                        "ep", "gp_ep"
                    )
                    gp.save(args["M"]["best_gp_path"])

                early_stop = 0
                # if best validation model, then testing
                test_model, test_callback = get_model(args)
                _ = test_epoch(args, test_model, test_callback, test_loader, writer, gp)

            else:
                early_stop += 1
                logging.info(
                    f"Early Stop Counter {early_stop} of {args['M']['max_early_stop']}."
                )

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


def get_list(cfg_path="origin.yaml", tag="base"):
    # generate train valid test list
    args = parse_yaml(cfg_path)
    setup_seed(args["M"]["seed"])

    log_dir = os.path.join(
        args["M"]["log_dir"],
        args["M"]["model"]
        + "_"
        + os.path.splitext(os.path.basename(cfg_path))[0]
        + "_"
        + tag,
    )

    if os.path.exists(log_dir):
        message = input(
            "[INFO] found duplicate directories, whether to overwrite (Y/N)"
        )
        if message.lower() == "y":
            print("[INFO] deleting the existing logs...")
            shutil.rmtree(log_dir)
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

    fold = []
    k = args["M"]["k"]
    for i in range(k):
        fold.append([v for v in range(0 + i, len(data_list), k)])

    # for i, (train_part, test_index) in enumerate(kf.split(data_list)):
    #     train_index, valid_index = split_train_valid(train_part, [0.8, 0.2])
    for i in range(k):
        test_part = fold[i]
        train_part = []
        for j in range(k):
            if j == i:
                continue
            else:
                train_part = train_part + fold[j]

        train_index, valid_index = split_train_valid(np.array(train_part), [0.7, 0.3])
        test_index = np.array(test_part)

        yamlname = (
            os.path.join(
                args["M"]["log_dir"],
                args["M"]["model"]
                + "_"
                + os.path.splitext(os.path.basename(cfg_path))[0]
                + "_"
                + tag,
                str(i),
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


def run_1_fold(cfg_path="origin.yaml", tag="base", index=0):
    get_list(cfg_path, tag)
    args = parse_yaml(cfg_path)
    setup_seed(args["M"]["seed"])
    log_dir = os.path.join(
        args["M"]["log_dir"],
        args["M"]["model"]
        + "_"
        + os.path.splitext(os.path.basename(cfg_path))[0]
        + "_"
        + tag,
    )
    assert os.path.exists(log_dir), "目录不存在, 请先运行get_list命令."
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
    pipline(args, train_list, valid_list, test_list)


def run_all(cfg_path="origin.yaml", tag="base"):
    get_list(cfg_path, tag)
    args = parse_yaml(cfg_path)
    setup_seed(args["M"]["seed"])
    log_dir = os.path.join(
        args["M"]["log_dir"],
        args["M"]["model"]
        + "_"
        + os.path.splitext(os.path.basename(cfg_path))[0]
        + "_"
        + tag,
    )
    assert os.path.exists(log_dir), "目录不存在, 请先运行get_list命令."

    for index in range(args["M"]["k"]):
        args["M"]["log_dir"] = os.path.join(log_dir, str(index))
        print(f"[INFO] loading C-V record {index}.yaml")
        data = parse_yaml(os.path.join(log_dir, str(index) + ".yaml"))
        train_list = data["train_list"]
        valid_list = data["valid_list"]
        test_list = data["test_list"]
        train_list = [os.path.join(args["D"]["data_dir"], i) for i in train_list]
        valid_list = [os.path.join(args["D"]["data_dir"], i) for i in valid_list]
        test_list = [os.path.join(args["D"]["data_dir"], i) for i in test_list]
        print(f"[INFO] start the pipline: {index}.yaml")
        # pipline(args, train_list, valid_list, test_list)
        print("[INFO] Query the idle GPU ...")
        os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
        memory_gpu = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        print(f"[INFO] GPU:{np.argmax(memory_gpu)} was selected")
        best_gpu = int(np.argmax(memory_gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        # os.system("rm tmp")  # 删除临时生成的 tmp 文件
        p = Process(target=pipline, args=(args, train_list, valid_list, test_list))
        p.start()
        time.sleep(5)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")  # good solution !!!!
    # fire.Fire(run_1_fold)
    fire.Fire(run_all)
