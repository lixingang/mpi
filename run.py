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

setup_seed(args.seed)

def logging_setting(args):
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.log_dir, f"run.log"), 
        filemode='w',
        format="%(asctime)s: %(message)s",
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


def _train(args, epoch, model, loader, gp=None):
    model.train()
    criterion = HEMLoss(0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.init_lr)
    scheduler = MultiStepLR(optimizer, **args.scheduler)
    losses = AverageMeter()
    y = []
    y_hat = []
    for fea, lbl in loader:
        img_data = fea[0]
        num_data = fea[1]
        _y = lbl["MPI3_fixed"].cuda()
        _y_hat, fea = model(img_data.cuda(), num_data.cuda())
        loss = criterion(_y_hat, _y)
        loss.backward()
        losses.update(loss)
        optimizer.step()
        optimizer.zero_grad()

        y.append(_y)
        y_hat.append(_y_hat)
        
        if gp:
            gp.append_training_params(
                fea.detach().cpu(), 
                lbl['year'],
                np.stack([lbl['lat'], lbl['lon']],-1),  #  60x2
                _y.cpu()
            )

    y = torch.cat(y, dim=0).detach()
    y_hat = torch.cat(y_hat, dim=0).detach()

    r2 = torchmetrics.functional.r2_score(y_hat, y).cpu().numpy()
    mse = torchmetrics.functional.mean_squared_error(y_hat, y).cpu().numpy()
    acc = {"loss":losses.avg(), "r2":r2, "mse":mse}
    logging.info(f"[train] epoch {epoch}/{args.epochs} r2={r2:.3f} mse={mse:.4f}")

    scheduler.step()
    if epoch%20==0:
        logging.info(f"epoch{epoch}, Current learning rate: {scheduler.get_last_lr()}")

    return model, acc, gp
 
def _validation(args, epoch, model, loader, gp=None):
    global early_stop
    with torch.no_grad():
        criterion = HEMLoss(0)
        losses = AverageMeter()
        y = []
        y_hat = []
        for fea, lbl in loader:
            img_data = fea[0]
            num_data = fea[1]
            _y = lbl["MPI3_fixed"].cuda()
            _y_hat, fea = model(img_data.cuda(), num_data.cuda())
            loss = criterion(_y_hat, _y)
            losses.update(loss)
            if gp:
                gp.append_testing_params(
                    fea.detach().cpu().numpy(), 
                    lbl['year'], 
                    np.stack([lbl['lat'], lbl['lon']],-1),
                    _y.cpu()
                )
            y.append(_y)
            y_hat.append(_y_hat)

        y = torch.cat(y, dim=0).detach()
        y_hat = torch.cat(y_hat, dim=0).detach() 

        if gp:
            y = gp.gp_run(
                epoch,
                model.state_dict()["fclayer.3.weight"].cpu(),
                model.state_dict()["fclayer.3.bias"].cpu(),
                args.log_dir,
            ).cuda()


        

        r2 = torchmetrics.functional.r2_score(y_hat, y).cpu().numpy()
        mse = torchmetrics.functional.mean_squared_error(y_hat, y).cpu().numpy()
        acc = {"loss":losses.avg(), "r2":r2, "mse":mse}

        
        logging.info(f"[valid] epoch {epoch}/{args.epochs} r2={acc['r2']:.3f} rmse={acc['mse']:.4f} ")
        
        
    return acc

def _test(args, epoch, model, loader, gp=None):
    with torch.no_grad():
        test_model = Net(args).cuda()

        # restore the parameters
        if args.best_weight_path is not None:
            test_model.load_state_dict(torch.load(args.best_weight_path))
            if gp:
                gp.restore(args.best_weight_path.replace("ep","gp_ep"))
        y = []
        y_hat = []
        
        for fea, lbl in loader:
            img_data = fea[0]
            num_data = fea[1]
            _y = lbl["MPI3_fixed"].cuda()
            _y_hat, fea = test_model(img_data.cuda(), num_data.cuda())
            if gp:
                gp.append_testing_params(
                    fea.detach().cpu().numpy(), 
                    lbl['year'], 
                    np.stack([lbl['lat'], lbl['lon']],-1),
                    _y.cpu()
                )
            y.append(_y)
            y_hat.append(_y_hat)

        y = torch.cat(y,dim=0).detach()
        y_hat = torch.cat(y_hat, dim=0).detach()

        if gp:
            y = gp.gp_run(
                epoch,
                test_model.state_dict()["fclayer.3.weight"].cpu(),
                test_model.state_dict()["fclayer.3.bias"].cpu(),
                args.log_dir,
            ).cuda()
        r2 = torchmetrics.functional.r2_score(y_hat, y).cpu().numpy()
        mse = torchmetrics.functional.mean_squared_error(y_hat, y).cpu().numpy()
        acc = {"r2":r2, "mse":mse}

        logging.info(f"[test] Testing with {args.best_weight_path}")
        logging.info(f"[test] r2={r2:.3f} mse={mse:.4f}")
        
        # model.load_state_dict(training_weight)

        return acc 


def run(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    logging_setting(args)
    logging.info(f"[Note] {args.note}")
    writer = SummaryWriter(log_dir=args.log_dir)
    
    data_list = np.array(glob.glob(f"{args.data_dir}/*"))
    train_list, valid_list, test_list = split_train_test(data_list, [0.7,0.15,0.15])
    with open(os.path.join(args.log_dir, 'train_valid_test.yaml'), 'w') as f:
        yaml.dump({"train_list":train_list.tolist(),"valid_list":valid_list.tolist(),"test_list":test_list.tolist(),}, f)


    logging.info(f"[DataSize] train,validate,test: {len(train_list)},{len(valid_list)},{len(test_list)}")

    train_loader = DataLoader(
        mpi_dataset(args, train_list), 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        mpi_dataset(args, valid_list),
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        mpi_dataset(args, test_list), 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=False,
    )

    model = Net(args).cuda()
    if args.restore_weight is not None:
        model.load_state_dict(torch.load(args.restore_weight))
    
    early_stop = 0 #  early stop

    gp = None
    if args.run_gp:
        gp = gp_model(sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01)

    for epoch in range(1, args.epochs+1):
        if gp:
            gp.clear_params()
        
        '''
        training
        '''
        model, acc, gp = _train(args, epoch, model, train_loader, gp)

        writer.add_scalar("Train/loss", acc['loss'], epoch)
        writer.add_scalar("Train/r2", acc['r2'], epoch)
        writer.add_scalar("Train/mse", acc['mse'], epoch)
        
        if epoch%5==0:
            '''
            validation
            '''
            acc = _validation(args, epoch, model, valid_loader, gp)
            if acc[args.best_acc["name"]]<args.best_acc["value"]:
                args.best_acc["value"] = float(acc[args.best_acc["name"]])
                os.mkdir(args.log_dir) if not os.path.exists(args.log_dir) else None   
                filename= f"ep{epoch}.pth"
                args.best_weight_path = os.path.join(args.log_dir, filename)
                torch.save(model.state_dict(), args.best_weight_path)
                if gp:
                    args.best_gp_path = args.best_weight_path.replace("ep","gp_ep")
                    gp.save(args.best_gp_path)

                early_stop = 0

            else:
                early_stop += 1
                logging.info(f"Early Stop Counter {early_stop} of {args.max_early_stop}.")

            if early_stop >= args.max_early_stop:
                break


            writer.add_scalar("Validate/loss", acc['loss'], epoch)
            writer.add_scalar("Validate/r2", acc['r2'], epoch)
            writer.add_scalar("Validate/mse", acc['mse'], epoch)
            

            '''
            testing
            '''
            acc = _test(args, epoch, model, test_loader, gp)
            writer.add_scalar("Test/r2", acc['r2'], epoch)
            writer.add_scalar("Test/mse", acc['mse'], epoch)
                
         
   
    '''
    final test
    '''
    acc = _test(args, epoch, model, test_loader, gp)
    writer.add_scalar("Test/r2", acc['r2'], epoch)
    writer.add_scalar("Test/mse", acc['mse'], epoch)
        
        # res = pd.DataFrame(res)
        # res.to_csv(f"{args.log_dir}/test_result.csv")

    
    save_args(args)
    return "OK"



print(run(args))
