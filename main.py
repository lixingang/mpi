# import office packages
import os
import sys
import yaml
from click import argument
import torch
import logging
import argparse
import glob
import time
import random
import datetime
import shutil
import math
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple

# import in-project packages
from Losses.loss import *
from Models import *
from Datasets.mpi_datasets import mpi_dataset
from Utils.base import parse_yaml, parse_log
from Utils.base import setup_seed, Meter, SaveOutput, split_train_test
from Utils.torchsummary import model_info
from Utils.LDS import get_lds_weights
'''
setup the random seed
'''


def args_setting(yamlpath, config_keys):
    args = parse_yaml(yamlpath)
    parser = argparse.ArgumentParser(description='Process some integers.')
    for key in config_keys:
        parser.add_argument(
            f'--{key}', type=type(args['M'][key]), default=None)
    arguments = parser.parse_args()
    arguments = vars(arguments)

    for key in config_keys:
        args['M'][key] = arguments[key]

    args['M']['log_dir'] = os.path.join(
        args['M']['log_dir'],
        args['M']['model']+"_"+args['M']['tag'],
        datetime.datetime.now().strftime('%b%d_%H-%M-%S'))

    return args


def logging_setting(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"run.log"),
        filemode='w',
        format="%(asctime)s: %(message)s",
        # format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.DEBUG
    )


def get_model(args):
    assert args['M']['model'].lower() in {'vit', 'mlp'}
    model = None
    callback = {}
    callback['last_fea'] = SaveOutput()
    callback['weights_fea'] = SaveOutput()
    if args['M']['model'].lower() == "vit":
        model = ViT(
            num_patches=len(args['D']['img_keys'])+len(args['D']['num_keys']),
            patch_dim=args['D']['in_channel'],
            num_classes=10,
            dim=args['VIT']['vit_dim'],
            depth=args['VIT']['vit_depth'],
            heads=args['VIT']['vit_heads'],
            mlp_dim=args['VIT']['vit_mlp_dim'],
            dropout=args['VIT']['vit_dropout'],
            emb_dropout=args['VIT']['vit_dropout']
        ).cuda()
        hook1 = model.mlp_head[0].register_forward_hook(callback['last_fea'])
        hook2 = model.mlp_head[1].register_forward_hook(callback['weights_fea'])
        
    elif args['M']['model'].lower() == 'mlp':
        model = MLP(
            len(args['D']['img_keys'])*args['D']['in_channel'],
            len(args['D']['num_keys']),
        ).cuda()

        hook1 = model.fclayer[2].register_forward_hook(callback['last_fea'])
        hook2 = model.fclayer[4].register_forward_hook(callback['weights_fea'])

    

    return model, callback


def _train_epoch(args, model, callback, epoch, loader, optimizer, gp=None, writer=None):
    model.train()
    losses, y, y_hat = Meter(), Meter(), Meter()

    for img, num, lbl, ind in loader:
        img = img.cuda()
        num = num.cuda()
        _y = lbl["MPI3_fixed"].cuda()
        ind = ind.float().cuda()

        aux = {"epoch":epoch, "label": _y}
        _y_hat, fea = model(img, num, aux)
        loss1 = weighted_huber_loss(_y_hat, _y, get_lds_weights(_y))

        loss2 = torch.nn.functional.mse_loss(ind, fea)

        loss = loss2
        loss.backward()
        losses.update(loss)
        optimizer.step()
        optimizer.zero_grad()
        y.update(_y)
        y_hat.update(_y_hat)

        if args['GP']['is_gp']:
            gp_training_params = {
                "feat": callback['last_fea'].data,
                "year": lbl['year'],
                "loc": np.stack([lbl['lat'], lbl['lon']], -1),
                "y": _y.cpu()
            }
            gp.append_training_params(**gp_training_params)

    r2 = torchmetrics.functional.r2_score(y_hat.cat(), y.cat()).numpy().item()
    rmse = math.sqrt(torchmetrics.functional.mean_squared_error(
        y_hat.cat(), y.cat()).numpy().item())
    acc = {"train/loss": losses.avg(), "train/r2": r2, "train/rmse": rmse}

    logging.info(
        f"[train] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} rmse={rmse:.4f}")
    if writer:
        writer.add_scalar("Train/loss", acc['train/loss'], epoch)
        writer.add_scalar("Train/r2", acc['train/r2'], epoch)
        writer.add_scalar("Train/rmse", acc['train/rmse'], epoch)

    if 1:
        y, y_hat = Meter(), Meter()
        feat = Meter()
        with torch.no_grad():
            for img, num, lbl, ind in loader:
                img = img.cuda()
                num = num.cuda()
                _y = lbl["MPI3_fixed"].cuda()
                other = {"epoch": epoch, "label": _y}
                _y_hat, _ = model(img, num, other)
                feat.update(callback['last_fea'].data)
                y.update(_y)
                y_hat.update(_y_hat)
        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(
            feat.cat(), y.cat(), epoch)
    return acc, gp


def _valid_epoch(args, model, callback, epoch, loader, gp=None, writer=None):
    global early_stop
    with torch.no_grad():
        model.eval()
        # criterion = HEMLoss(0)
        losses, y, y_hat = Meter(), Meter(), Meter()
        for img, num, lbl, ind in loader:
            img = img.cuda()
            num = num.cuda()
            _y = lbl["MPI3_fixed"].cuda()
            _y_hat, _ = model(img, num)
            loss = weighted_mse_loss(_y_hat, _y, get_lds_weights(_y))
            losses.update(loss)
            if gp:
                gp.append_testing_params(
                    callback['last_fea'].data,
                    lbl['year'],
                    np.stack([lbl['lat'], lbl['lon']], -1),
                )
            y.update(_y)
            y_hat.update(_y_hat)

        if gp:
            y_hat = gp.gp_run(
                model.state_dict()["fclayer.3.weight"].cpu(),
                model.state_dict()["fclayer.3.bias"].cpu(),
            ).cuda()

        r2 = torchmetrics.functional.r2_score(
            y_hat.cat(), y.cat()).numpy().item()
        rmse = math.sqrt(torchmetrics.functional.mean_squared_error(
            y_hat.cat(), y.cat()).numpy().item())
        acc = {"valid/loss": losses.avg(), "valid/r2": r2, "valid/rmse": rmse}

        logging.info(
            f"[valid] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} rmse={rmse:.4f} ")
        if writer:
            writer.add_scalar("Valid/loss", acc['valid/loss'], epoch)
            writer.add_scalar("Valid/r2", acc['valid/r2'], epoch)
            writer.add_scalar("Valid/rmse", acc['valid/rmse'], epoch)
    return acc


def _test_epoch(args, model, callback, epoch, loader, gp=None, writer=None):
    with torch.no_grad():
        model.eval()

        # restore the parameters
        model.load_state_dict(torch.load(args['M']['best_weight_path']))
        if gp:
            gp.restore(args['M']['best_gp_path'])

        y, y_hat, names, lons, lats = Meter(), Meter(), Meter(), Meter(), Meter()
        w_feats, w_ind = Meter(), Meter()
        for img, num, lbl, ind in loader:
            img = img.cuda()
            num = num.cuda()
            _y = lbl["MPI3_fixed"].cuda()
            _y_hat, _ = model(img, num)
            if args['GP']['is_gp']:
                gp.append_testing_params(
                    callback['last_fea'].data,
                    lbl['year'],
                    np.stack([lbl['lat'], lbl['lon']], -1),
                )
            y.update(_y)
            y_hat.update(_y_hat)
            names.update(lbl['name'])
            lons.update(lbl['lon'])
            lats.update(lbl['lat'])
            w_feats.update(callback['weights_fea'].data)
            w_ind.update(ind)
        w_feats = w_feats.cat().cpu().detach().numpy()
        w_ind = w_ind.cat().cpu().detach().numpy()
        np.savetxt(os.path.join(
            args['M']['log_dir'], "weight_features.csv"), w_feats, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(
            args['M']['log_dir'], "weight_indicator.csv"), w_ind, delimiter=',', fmt='%.3f')
        if args['GP']['is_gp']:
            y_hat = gp.gp_run(
                model.state_dict()["fclayer.3.weight"].cpu(),
                model.state_dict()["fclayer.3.bias"].cpu(),
            ).cuda()
        r2 = torchmetrics.functional.r2_score(
            y_hat.cat(), y.cat()).numpy().item()
        rmse = math.sqrt(torchmetrics.functional.mean_squared_error(
            y_hat.cat(), y.cat()).numpy().item())
        acc = {"test/r2": r2, "test/rmse": rmse}

        logging.info(f"[test] Testing with {args['M']['best_weight_path']}")
        logging.info(f"[test] r2={r2:.3f} rmse={rmse:.4f}")
        if writer:
            writer.add_scalar("Test/r2", acc['test/r2'], epoch)
            writer.add_scalar("Test/rmse", acc['test/rmse'], epoch)

        df = pd.DataFrame({"name": names.cat(), "y": y.cat(
        ), "y_hat": y_hat.cat(), "lon": lons.cat(), "lat": lats.cat(), })
        df.to_csv(os.path.join(args['M']['log_dir'], "predict.csv"))

        return acc


def run():
    # 开始训练前的处理，包含配置文件、随机种子、显卡、tb定义
    args = args_setting("config.yaml", ['seed', 'gpu', 'tag', 'model'])
    setup_seed(args['M']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = args['M']['gpu']
    logging_setting(args['M']['log_dir'])
    writer = SummaryWriter(log_dir=args['M']['log_dir'])

    # 划分数据集并保存、得到dataloader
    data = pd.read_csv(args['D']['source'])
    data_list = data['name'].tolist()
    mpi_list = data['MPI3_fixed'].tolist()
    train_list, valid_list, test_list = split_train_test(
        data_list, [0.7, 0.15, 0.15])

    # 保存测试、验证、测试集到文件
    with open(os.path.join(args['M']['log_dir'], 'train_valid_test.yaml'), 'w') as f:
        yaml.dump({"train_list": train_list,
                  "valid_list": valid_list, "test_list": test_list}, f)

    logging.info(
        f"[DataSize] train,validate,test: {len(train_list)},{len(valid_list)},{len(test_list)}")

    # 定义 data loader
    train_loader = DataLoader(
        mpi_dataset(args, train_list),
        batch_size=args['M']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        mpi_dataset(args, valid_list),
        batch_size=args['M']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        mpi_dataset(args, test_list),
        batch_size=args['M']['batch_size'],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    # 获取model 并打印模型结构到文件
    train_model, train_callback = get_model(args)
    print(train_model, file=open(
        os.path.join(args['M']['log_dir'], "model.txt"), "a"))
    
    # 定义优化器
    optimizer = torch.optim.Adam(filter(
        lambda p: p.requires_grad, train_model.parameters()), lr=args['M']['init_lr'])
    scheduler = MultiStepLR(optimizer, **args['M']['scheduler'])

    # 如果有，则恢复训练权重
    if args['M']['restore_weight'] is not None:
        train_model.load_state_dict(torch.load(args['M']['restore_weight']))

    early_stop = 0  # early stop
    gp = None
    for epoch in range(1, args['M']['epochs']+1):
        if args['GP']['is_gp']:
            gp = gp_model(sigma=1, r_loc=2.5, r_year=3,
                          sigma_e=0.32, sigma_b=0.01)
            gp.clear_params()

        # training
        if epoch % 1 == 0:
            _, gp = _train_epoch(args, train_model, train_callback, epoch,
                                 train_loader, optimizer, gp, writer)

        # validation
        if epoch % 5 == 0:
            valid_model, callback = get_model(args)
            valid_model.load_state_dict(train_model.state_dict())
            valid_acc = _valid_epoch(
                args, valid_model, callback, epoch,  valid_loader, gp, writer)
            if valid_acc["valid/rmse"] < args['M']['best_acc']:
                args['M']['best_acc'] = valid_acc["valid/rmse"]
                args['M']['best_weight_path'] = os.path.join(
                    args['M']['log_dir'], f"ep{epoch}.pth")
                torch.save(train_model.state_dict(),
                           args['M']['best_weight_path'])
                if args['GP']['is_gp']:
                    args['M']['best_gp_path'] = args['M']['best_weight_path'].replace(
                        "ep", "gp_ep")
                    gp.save(args['M']['best_gp_path'])

                early_stop = 0

            else:
                early_stop += 1
                logging.info(
                    f"Early Stop Counter {early_stop} of {args['M']['max_early_stop']}.")

            if early_stop >= args['M']['max_early_stop']:
                break

        # testing
        if epoch % 5 == 0:
            test_model, callback = get_model(args)
            _ = _test_epoch(args, test_model, callback,
                            epoch, test_loader, gp, writer)

        scheduler.step()

        if epoch % 20 == 0:
            logging.info(
                f"epoch{epoch}, Current learning rate: {scheduler.get_last_lr()}")

    '''
    final test
    '''
    test_model, callback = get_model(args)
    _ = _test_epoch(args, test_model, callback,
                    epoch, test_loader, gp, writer)

    # save args
    os.makedirs(args['M']['log_dir'], exist_ok=True)
    with open(os.path.join(args['M']['log_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(args, f)
    return "OK"


print(run())
