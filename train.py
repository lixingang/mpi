# import office packages
import os,sys,yaml
import torch,logging,argparse,glob,time,random,datetime,shutil
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim.lr_scheduler import StepLR,LambdaLR,MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import numpy as np
import pandas as pd
# import in-project packages
from Losses.loss import HEMLoss,CenterLoss
from Models.network import Net
from Models.gp import gp_model
from Datasets.mpi_datasets import mpi_dataset
from Utils.AverageMeter import AverageMeter
from Utils.clock import clock,Timer
from Utils.setup_seed import setup_seed
from Utils.ParseYAML import ParseYAML


config = ParseYAML("config.yaml")
parser = argparse.ArgumentParser(description='Process some integers.')
for key in config:
    parser.add_argument(f'--{key}', default=config[key],type=type(config[key]))
parser.add_argument(f'--note', default="",type=str)
args = parser.parse_args()
args.log_dir = os.path.join("Logs", args.model_name, datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
# for arg in vars(args):
#     print( arg, getattr(args, arg))
setup_seed(args.seed)

def logging_setting(args):
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, f"run.log"), 
        filemode='w',
        format="%(asctime)s %(levelname)s: %(message)s",
        # format="%(asctime)s %(levelname)s: \033[0;33m%(message)s\033[0m",
        datefmt="%Y%m%d %H:%M:%S",
        level=logging.DEBUG 
    )

def save_args(args):
    
    argsDict = args.__dict__
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(argsDict, f)

def split_train_test(data_list, ratio=[0.6,0.2,0.2]):
    idx = list(range(len(data_list)))
    random.shuffle(idx)
    assert len(ratio)>=2 and len(ratio)<=3
    assert np.sum(np.array(ratio))==1.0
    slice1 = int(len(idx)*ratio[0])
    slice2 = int(len(idx)*(ratio[1]+ratio[0]))
    if len(ratio)==2:
        return data_list[:slice1],data_list[slice1:slice2]
    else:
        return data_list[:slice1],data_list[slice1:slice2],data_list[slice2:]


def train(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    logging_setting(args)
    logging.info(f"[Note] {args.note}")
    writer = SummaryWriter(log_dir=args.log_dir)
    
    data_list = np.array(glob.glob(f"{args.data_dir}/*"))
    train_list, valid_list, test_list = split_train_test(data_list, [0.7,0.15,0.15])
    with open(os.path.join(args.log_dir, 'train_valid_test.yaml'), 'w') as f:
        yaml.dump({"train_list":train_list.tolist(),"valid_list":valid_list.tolist(),"test_list":test_list.tolist(),}, f)


    logging.info(f"[DataSize] train,validate,test: {len(train_list)},{len(valid_list)},{len(test_list)}")

    train_dataset = mpi_dataset(args, train_list)
    valid_dataset = mpi_dataset(args, valid_list)
    test_dataset = mpi_dataset(args, test_list)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
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

    model = Net(args).cuda()
    if args.restore_weight is not None:
        model.load_state_dict(torch.load(args.restore_weight))
    
    criterion = HEMLoss(0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.init_lr)
    scheduler = MultiStepLR(optimizer, **args.scheduler)
    metrics = {
        "r2": torchmetrics.R2Score().cuda(),
        "mape": torchmetrics.MeanAbsolutePercentageError().cuda(),
        "mse": torchmetrics.MeanSquaredError().cuda(),
    }
    
    early_stop = 0 #  early stop
    gp = gp_model(sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01)
    for epoch in range(1, args.epochs+1):
        gp.clear_training_params()
        gp.clear_testing_params()
        model.train()
        _ = [metrics[k].reset() for k in metrics.keys()]
        losses = AverageMeter()
        for fea, lbl in train_loader:
            img_data = fea[0]
            num_data = fea[1]
            y = lbl["MPI3_fixed"].cuda()
            y_hat, fea = model(img_data.cuda(), num_data.cuda())
            loss = criterion(y_hat, y)
            loss.backward()
            losses.update(loss)
            optimizer.step()
            optimizer.zero_grad()
            acc = {k: metrics[k](y_hat, y) for k in metrics.keys()}
            gp.update_training_params(
                fea.detach().cpu(), 
                lbl['year'],
                np.stack([lbl['lat'], lbl['lon']],-1),  #  60x2
                y.cpu()
            )
            

        acc = {k: metrics[k].compute() for k in metrics.keys()}
        writer.add_scalar("Train/loss", losses.avg(), epoch)
        writer.add_scalar("Train/r2", acc['r2'], epoch)
        writer.add_scalar("Train/mse", acc['mse'], epoch)
        logging.info(f"[train] epoch {epoch}/{args.epochs} r2={acc['r2']:.3f} mse={acc['mse']:.4f} mape:{acc['mape']:.3f}")

        if epoch%3==0:
            with torch.no_grad():
                _ = [metrics[k].reset() for k in metrics.keys()]
                losses = AverageMeter()
                for fea, lbl in valid_loader:
                    img_data = fea[0]
                    num_data = fea[1]
                    y = lbl["MPI3_fixed"].cuda()
                    y_hat, fea = model(img_data.cuda(), num_data.cuda())
                    loss = criterion(y_hat, y)
                    losses.update(loss)
                    acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
                    
                acc = {k: metrics[k].compute() for k in metrics.keys()}
                if acc[args.best_acc["name"]]<args.best_acc["value"]:
                    args.best_acc["value"] = float(acc[args.best_acc["name"]])
                    os.mkdir(args.log_dir) if not os.path.exists(args.log_dir) else None   
                    filename= f"epoch{epoch}_{args.best_acc['name']}_{args.best_acc['value']:.4f}.pth"
                    args.best_weight_path = os.path.join(args.log_dir, filename)
                    # best_weight = model.state_dict()
                    torch.save(model.state_dict(), args.best_weight_path)
                    early_stop = 0
                else:
                    early_stop += 1
                    logging.info(f"Early Stop Counter {early_stop} of {args.max_early_stop}.")
                    if early_stop >= args.max_early_stop:
                        break
                writer.add_scalar("Validate/loss", losses.avg(), epoch)
                writer.add_scalar("Validate/r2", acc['r2'], epoch)
                writer.add_scalar("Validate/mse", acc['mse'], epoch)
                logging.info(f"[valid] epoch {epoch}/{args.epochs} r2={acc['r2']:.3f} rmse={acc['mse']:.4f} mape={acc['mape']:.3f}")
                logging.info(f"epoch{epoch}, Current learning rate: {scheduler.get_last_lr()}")
        
        

        if epoch%10==0:
            gp.clear_testing_params()
            with torch.no_grad():
                training_weight = model.state_dict()
                if args.best_weight_path is not None:
                    model.load_state_dict(torch.load(args.best_weight_path))
                _ = [metrics[k].reset() for k in metrics.keys()]
                test_y = []
                pred_y = []
                for fea, lbl in test_loader:
                    img_data = fea[0]
                    num_data = fea[1]
                    y = lbl["MPI3_fixed"].cuda()
                    y_hat, fea = model(img_data.cuda(), num_data.cuda())
                    gp.update_testing_params(
                        fea.detach().cpu().numpy(), 
                        lbl['year'], 
                        np.stack([lbl['lat'], lbl['lon']],-1),
                        y.cpu()
                    )
                    test_y.append(y)
                    pred_y.append(y_hat)

                test_y = torch.cat(test_y, dim=0)
                pred_y = torch.cat(pred_y, dim=0)
                y_gp = gp.gp_run(
                    epoch,
                    model.state_dict()["fclayer.3.weight"].cpu(),
                    model.state_dict()["fclayer.3.bias"].cpu(),
                    args.log_dir,
                ).cuda()
                r2 = torchmetrics.functional.r2_score(pred_y, test_y).cpu().numpy()
                mse = torchmetrics.functional.mean_squared_error(pred_y, test_y).cpu().numpy()
                r2_gp = torchmetrics.functional.r2_score(y_gp, test_y).cpu().numpy()
                mse_gp = torchmetrics.functional.mean_squared_error(y_gp, test_y).cpu().numpy()

                writer.add_scalar("Test/r2", r2, epoch)
                writer.add_scalar("Test/r2_gp", r2_gp, epoch)
                writer.add_scalar("Test/mse", mse, epoch)
                writer.add_scalar("Test/mse_gp", mse_gp, epoch)
                logging.info(f"[test] Testing with {args.best_weight_path}")
                logging.info(f"[test] r2_old={r2:.3f} mse_old={mse:.4f}")
                logging.info(f"[test] r2={r2_gp:.3f} mse={mse_gp:.4f}")
                model.load_state_dict(training_weight)
         
                

                
        scheduler.step()


    gp.clear_testing_params()
    with torch.no_grad():
        res = {"name":[],"y":[],"y_hat":[]}
        logging.info(f"[test] Finish training or trigger early stop")
        if args.best_weight_path is not None:
            model.load_state_dict(torch.load(args.best_weight_path))
        test_y = []
        pred_y = []
        for fea, lbl in test_loader:
            img_data = fea[0]
            num_data = fea[1]
            y = lbl["MPI3_fixed"].cuda()
            y_hat, fea = model(img_data.cuda(), num_data.cuda())
            res["name"].extend(lbl["name"])
            res["y"].extend(y.cpu())
            res["y_hat"].extend(y_hat.cpu())
            gp.update_testing_params(
                fea.detach().cpu().numpy(), 
                lbl['year'], 
                np.stack([lbl['lat'], lbl['lon']],-1),
                y.cpu()
            )
            test_y.append(y)
            pred_y.append(y_hat)

        test_y = torch.cat(test_y, dim=0)
        pred_y = torch.cat(pred_y, dim=0)
        y_gp = gp.gp_run(
            epoch,
            model.state_dict()["fclayer.3.weight"].cpu(),
            model.state_dict()["fclayer.3.bias"].cpu(),
            args.log_dir,
        ).cuda()
        r2 = torchmetrics.functional.r2_score(pred_y, test_y).cpu().numpy()
        mse = torchmetrics.functional.mean_squared_error(pred_y, test_y).cpu().numpy()
        r2_gp = torchmetrics.functional.r2_score(y_gp, test_y).cpu().numpy()
        mse_gp = torchmetrics.functional.mean_squared_error(y_gp, test_y).cpu().numpy()
        
        writer.add_scalar("Test/r2", r2, epoch)
        writer.add_scalar("Test/r2_gp", r2_gp, epoch)
        writer.add_scalar("Test/mse", mse, epoch)
        writer.add_scalar("Test/mse_gp", mse_gp, epoch)
        logging.info(f"[test] Testing with {args.best_weight_path}")
        logging.info(f"[test] r2_old={r2:.3f} rmse_old={mse:.4f}")
        logging.info(f"[test] r2={r2_gp:.3f} mse={mse_gp:.4f}")
        # res = pd.DataFrame(res)
        # res.to_csv(f"{args.log_dir}/test_result.csv")

    
    save_args(args)
    return "OK"



print(train(args))
