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
from Utils.parse import parse_yaml, parse_log
from Utils.base import setup_seed, Meter, SaveOutput
from Utils.torchsummary import model_info
from Utils.LDS import get_lds_weights
'''
setup the random seed
'''


def args_setting(path, config_keys):
    args = parse_yaml(path)
    parser = argparse.ArgumentParser(description='Process some integers.')
    for key in config_keys:
        parser.add_argument(
            f'--{key}', type=type(args['M'][key]), default=None)
    arguments = parser.parse_args()
    arguments = vars(arguments)

    for key in config_keys:
        args['M'][key] = arguments[key]

    args['M']['log_dir'] = os.path.join(
        "Logs",
        args['D']['name']+"_"+args['M']['model']+"_"+args['M']['tag'],
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


def save_args(args):
    os.makedirs(args['M']['log_dir'], exist_ok=True)
    with open(os.path.join(args['M']['log_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(args, f)


def split_train_test(data_list, ratio=[0.6, 0.2, 0.2]):
    idx = list(range(len(data_list)))
    random.shuffle(idx)
    assert len(ratio) >= 2 and len(ratio) <= 3
    assert np.sum(np.array(ratio)) == 1.0
    slice1 = int(len(idx)*ratio[0])
    slice2 = int(len(idx)*(ratio[1]+ratio[0]))
    if len(ratio) == 2:
        return data_list[:slice1], data_list[slice1:slice2]
    else:
        return data_list[:slice1], data_list[slice1:slice2], data_list[slice2:]


def get_model(args):
    assert args['M']['model'].lower() in {'vit', 'mlp'}
    model = None
    if args['M']['model'].lower() == "vit":
        model = ViT(
            num_patches=len(args['D']['img_keys'])+len(args['D']['num_keys']),
            patch_dim=args['D']['in_channel'],
            num_classes=1,
            dim=args['VIT']['vit_dim'],
            depth=args['VIT']['vit_depth'],
            heads=args['VIT']['vit_heads'],
            mlp_dim=args['VIT']['vit_mlp_dim'],
            dropout=args['VIT']['vit_dropout'],
            emb_dropout=args['VIT']['vit_dropout']
        ).cuda()
    elif args['M']['model'].lower() == 'mlp':
        model = MLP(
            len(args['D']['img_keys'])*args['D']['in_channel'],
            len(args['D']['num_keys']),
        ).cuda()

    return model


def _train_epoch(args, model, epoch, loader, optimizer, gp=None, writer=None):
    model.train()
    losses, y, y_hat = Meter(), Meter(), Meter()
    last_fea = SaveOutput()
    hooks = model.fclayer[2].register_forward_hook(last_fea)
    for img, num, lbl in loader:
        img = img.cuda()
        num = num.cuda()
        _y = lbl["MPI3_fixed"].cuda()
        aux = {}
        _y_hat = model(img, num, aux)
        loss = weighted_huber_loss(_y_hat, _y, get_lds_weights(_y))
        loss.backward()
        losses.update(loss)
        optimizer.step()
        optimizer.zero_grad()
        y.update(_y)
        y_hat.update(_y_hat)
        if args['GP']['is_gp']:
            gp_training_params = {
                "feat": last_fea[-1],
                "year": lbl['year'],
                "loc": np.stack([lbl['lat'], lbl['lon']], -1),
                "y": _y.cpu()
            }
            gp.append_training_params(**gp_training_params)

    r2 = torchmetrics.functional.r2_score(y_hat.cat(), y.cat()).numpy().item()
    mse = torchmetrics.functional.mean_squared_error(
        y_hat.cat(), y.cat()).numpy().item()
    acc = {"train/loss": losses.avg(), "train/r2": r2, "train/mse": mse}
    last_fea.clear()
    logging.info(
        f"[train] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} mse={mse:.4f}")
    if writer:
        writer.add_scalar("Train/loss", acc['train/loss'], epoch)
        writer.add_scalar("Train/r2", acc['train/r2'], epoch)
        writer.add_scalar("Train/mse", acc['train/mse'], epoch)

    if 1:
        y, y_hat = Meter(), Meter()
        with torch.no_grad():
            for img, num, lbl in loader:
                img = img.cuda()
                num = num.cuda()
                _y = lbl["MPI3_fixed"].cuda()
                other = {"epoch": epoch, "label": _y}
                _y_hat = model(img, num, other)

                y.update(_y)
                y_hat.update(_y_hat)

        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(
            last_fea.cat_then_clear(), y.cat(), epoch)

    return acc, gp


def _valid_epoch(args, model, epoch, loader, gp=None, writer=None):
    global early_stop
    last_fea = SaveOutput()
    hooks = model.fclayer[2].register_forward_hook(last_fea)
    with torch.no_grad():
        model.eval()
        # criterion = HEMLoss(0)
        losses, y, y_hat = Meter(), Meter(), Meter()
        for img, num, lbl in loader:
            img = img.cuda()
            num = num.cuda()
            _y = lbl["MPI3_fixed"].cuda()
            _y_hat = model(img, num)
            loss = weighted_mse_loss(_y_hat, _y, get_lds_weights(_y))
            losses.update(loss)
            if gp:
                gp.append_testing_params(
                    last_fea[-1],
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
        mse = torchmetrics.functional.mean_squared_error(
            y_hat.cat(), y.cat()).numpy().item()
        acc = {"valid/loss": losses.avg(), "valid/r2": r2, "valid/mse": mse}

        logging.info(
            f"[valid] epoch {epoch}/{args['M']['epochs']} r2={r2:.3f} mse={mse:.4f} ")
        if writer:
            writer.add_scalar("Valid/loss", acc['valid/loss'], epoch)
            writer.add_scalar("Valid/r2", acc['valid/r2'], epoch)
            writer.add_scalar("Valid/mse", acc['valid/mse'], epoch)
    return acc


def _test_epoch(args, model, epoch, loader, gp=None, writer=None):
    with torch.no_grad():
        model.eval()
        last_fea = SaveOutput()
        hooks = model.fclayer[2].register_forward_hook(last_fea)
        # restore the parameters
        model.load_state_dict(torch.load(args['M']['best_weight_path']))
        if gp:
            gp.restore(args['M']['best_gp_path'])

        y, y_hat, names, lons, lats = Meter(), Meter(), Meter(), Meter(), Meter()
        for img, num, lbl in loader:
            img = img.cuda()
            num = num.cuda()
            _y = lbl["MPI3_fixed"].cuda()
            _y_hat = model(img, num)
            if args['GP']['is_gp']:
                gp.append_testing_params(
                    last_fea[-1],
                    lbl['year'],
                    np.stack([lbl['lat'], lbl['lon']], -1),
                )
            y.update(_y)
            y_hat.update(_y_hat)
            names.update(lbl['name'])
            lons.update(lbl['lon'])
            lats.update(lbl['lat'])

        if args['GP']['is_gp']:
            y_hat = gp.gp_run(
                model.state_dict()["fclayer.3.weight"].cpu(),
                model.state_dict()["fclayer.3.bias"].cpu(),
            ).cuda()
        r2 = torchmetrics.functional.r2_score(
            y_hat.cat(), y.cat()).numpy().item()
        mse = torchmetrics.functional.mean_squared_error(
            y_hat.cat(), y.cat()).numpy().item()
        acc = {"test/r2": r2, "test/mse": mse}

        logging.info(f"[test] Testing with {args['M']['best_weight_path']}")
        logging.info(f"[test] r2={r2:.3f} mse={mse:.4f}")
        if writer:
            writer.add_scalar("Test/r2", acc['test/r2'], epoch)
            writer.add_scalar("Test/mse", acc['test/mse'], epoch)

        df = pd.DataFrame({"name": names.cat(), "y": y.cat(
        ), "y_hat": y_hat.cat(), "lon": lons.cat(), "lat": lats.cat(), })
        df.to_csv(os.path.join(args['M']['log_dir'], "predict.csv"))

        return acc


def run():
    args = args_setting("config.yaml", ['seed', 'gpu', 'tag', 'model'])
    setup_seed(args['M']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = args['M']['gpu']
    logging_setting(args['M']['log_dir'])
    writer = SummaryWriter(log_dir=args['M']['log_dir'])

    data_list = np.array(glob.glob(f"Data/{args['D']['name']}/*"))
    train_list, valid_list, test_list = split_train_test(
        data_list, [0.7, 0.15, 0.15])
    with open(os.path.join(args['M']['log_dir'], 'train_valid_test.yaml'), 'w') as f:
        yaml.dump({"train_list": train_list.tolist(
        ), "valid_list": valid_list.tolist(), "test_list": test_list.tolist(), }, f)

    logging.info(
        f"[DataSize] train,validate,test: {len(train_list)},{len(valid_list)},{len(test_list)}")

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

    train_model = get_model(args)
    model_str, params_info = model_info(
        train_model,
        [
            (args['M']['batch_size'], args['D']
             ['in_channel'], len(args['D']['img_keys'])),
            (args['M']['batch_size'], len(args['D']['num_keys']))
        ]
    )
    with open(os.path.join(args['M']['log_dir'], "model_summary.log"), "w") as f:
        f.write(model_str)

    optimizer = torch.optim.Adam(filter(
        lambda p: p.requires_grad, train_model.parameters()), lr=args['M']['init_lr'])
    scheduler = MultiStepLR(optimizer, **args['M']['scheduler'])

    if args['M']['restore_weight'] is not None:
        train_model.load_state_dict(torch.load(args['M']['restore_weight']))

    early_stop = 0  # early stop
    gp = None
    for epoch in range(1, args['M']['epochs']+1):
        if args['GP']['is_gp']:
            gp = gp_model(sigma=1, r_loc=2.5, r_year=3.,
                          sigma_e=0.32, sigma_b=0.01)
            gp.clear_params()

        # training
        if epoch % 1 == 0:
            _, gp = _train_epoch(args, train_model, epoch,
                                 train_loader, optimizer, gp, writer)

        # validation
        if epoch % 5 == 0:
            valid_model = get_model(args)
            valid_model.load_state_dict(train_model.state_dict())
            valid_acc = _valid_epoch(
                args, valid_model, epoch,  valid_loader, gp, writer)
            if valid_acc["valid/mse"] < args['M']['best_acc']:
                args['M']['best_acc'] = valid_acc["valid/mse"]
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
            test_model = get_model(args)
            _ = _test_epoch(args, test_model, epoch, test_loader, gp, writer)

        scheduler.step()

        if epoch % 20 == 0:
            logging.info(
                f"epoch{epoch}, Current learning rate: {scheduler.get_last_lr()}")

    '''
    final test
    '''
    _ = _test_epoch(args, epoch, test_loader, gp, writer)

    save_args(args)
    return "OK"


print(run())
