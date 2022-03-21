# import office packages
import os,sys,yaml
import torch,logging,argparse,h5py,glob,time,random,datetime,shutil
from torch.utils.data import DataLoader,ConcatDataset
from torch.optim.lr_scheduler import StepLR,LambdaLR,MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import in-project packages
from Losses.loss import HEMLoss,CenterLoss
from Models.network import Net
from Models.mpi_datasets import mpi_dataset
sys.path.append("./Metrics")
import Metrics.torchmetrics as torchmetrics
from Utils.AverageMeter import AverageMeter
from Utils.clock import clock,Timer
from Utils.setup_seed import setup_seed
from Utils.ConfigDict import ConfigDict

config = ConfigDict("config.yaml")
config['log_dir'] = os.path.join("Logs", datetime.datetime.now().strftime('%b%d_%H-%M-%S'))
parser = argparse.ArgumentParser(description='Process some integers.')
for key in config:
    parser.add_argument(f'--{key}', default=config[key],type=type(config[key]))
parser.add_argument(f'--note', default="",type=str)
args = parser.parse_args()
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

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    logging_setting(args)
    logging.info(f"[Note] {args.note}")
    writer = SummaryWriter(log_dir=args.log_dir)
    logging.info(f"The Dataset: {[h5path for h5path in args.h5_dir]}")
    ds = ConcatDataset([mpi_dataset(args, h5path) for h5path in args.h5_dir])

    train_size = int(len(ds) * 0.7)
    validate_size = int(len(ds) * 0.15)
    test_size = len(ds) - validate_size - train_size
    logging.info(f"[DataSize] train,validate,test: {train_size},{validate_size},{test_size}")

    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, validate_size, test_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    validate_loader = DataLoader(
        validate_dataset, 
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
    for epoch in range(1, args.epochs+1):
        model.train()
        _ = [metrics[k].reset() for k in metrics.keys()]
        losses = AverageMeter()
        for y, fea_img, fea_num in train_loader:
            y = y.cuda()
            y_hat = model(fea_img.cuda(), fea_num.cuda())
            loss = criterion(y_hat, y)
            loss.backward()
            losses.update(loss)
            optimizer.step()
            optimizer.zero_grad()
            acc = {k: metrics[k](y_hat, y) for k in metrics.keys()}
        acc = {k: metrics[k].compute() for k in metrics.keys()}
        writer.add_scalar("Train/loss", losses.avg(), epoch)
        writer.add_scalar("Train/r2", acc['r2'], epoch)
        writer.add_scalar("Train/mse", acc['mse'], epoch)
        logging.info(f"[train] epoch {epoch}/{args.epochs} r2={acc['r2']:.3f} rmse={acc['mse']:.4f} mape:{acc['mape']:.3f}")

        if epoch%3==0:
            with torch.no_grad():
                _ = [metrics[k].reset() for k in metrics.keys()]
                losses = AverageMeter()
                for y, fea_img, fea_num in validate_loader:
                    y = y.cuda()
                    y_hat = model(fea_img.cuda(), fea_num.cuda())
                    loss = criterion(y_hat, y)
                    losses.update(loss)
                    acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
                if acc[args.best_acc["name"]]<args.best_acc["value"]:
                    args.best_acc["value"] = float(acc[args.best_acc["name"]])
                    os.mkdir(args.log_dir) if not os.path.exists(args.log_dir) else None   
                    filename= f"{args.model_name}_epoch{epoch}_{args.best_acc['name']}_{args.best_acc['value']:.4f}.pth"
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
                logging.info(f"[valid] epoch {epoch}/{args.epochs} r2={acc['r2']:.3f} rmse={acc['mse']:.4f} mape:{acc['mape']:.3f}")
                logging.info(f"epoch{epoch}, Current learning rate: {scheduler.get_last_lr()}")
            
        scheduler.step()
        
        
        if epoch%10==0:
            with torch.no_grad():
                training_weight = model.state_dict()
                if args.best_weight_path is not None:
                    # logging.info(f"[test] loading best weight: {args.best_weight_path}")
                    model.load_state_dict(torch.load(args.best_weight_path))
                _ = [metrics[k].reset() for k in metrics.keys()]
                for y, fea_img, fea_num in test_loader:
                    y = y.cuda()
                    y_hat = model(fea_img.cuda(), fea_num.cuda())
                    acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
                writer.add_scalar("Test/r2", acc['r2'], epoch)
                writer.add_scalar("Test/mse", acc['mse'], epoch)
                logging.info(f"[test] Testing with {args.best_weight_path}")
                logging.info(f"[test] r2={acc['r2']:.3f} rmse={acc['mse']:.4f} mape:{acc['mape']:.3f}")
                model.load_state_dict(training_weight)
    
    # 
    with torch.no_grad():
        logging.info(f"[test] Finish training or trigger early stop")
        if args.best_weight_path is not None:
            # logging.info(f"[test] loading best weight: {args.best_weight_path}")
            model.load_state_dict(torch.load(args.best_weight_path))
        _ = [metrics[k].reset() for k in metrics.keys()]
        for y, fea_img, fea_num in test_loader:
            y = y.cuda()
            y_hat = model(fea_img.cuda(), fea_num.cuda())
            acc = {key: metrics[key](y_hat, y) for key in metrics.keys()}
        writer.add_scalar("Test/r2", acc['r2'], epoch)
        writer.add_scalar("Test/mse", acc['mse'], epoch)
        logging.info(f"[test] Testing with {args.best_weight_path}")
        logging.info(f"[test] r2={acc['r2']:.3f} rmse={acc['mse']:.4f} mape:{acc['mape']:.3f}")
    
    save_args(args)
    return "OK"



print(train(args))
