import h5py
from Models.mpi_datasets import mpi_dataset
from Utils.clock import clock
import torch
with h5py.File("/mnt/d12t/mpi/Data/fold0.h5") as f:
    
    DATASET = mpi_dataset(f)
    dataloaders = torch.utils.data.DataLoader(
                dataset=DATASET,
                batch_size=5,
                shuffle=True,
                num_workers=1,
                pin_memory=False,
                drop_last=False)
    for i in dataloaders:
        a,b = i[0],i[1]
        print(":",a.shape,b.shape)


import os,sys
sys.path.extend([os.listdir("./")])
import torchmetrics
import torch
metric = torchmetrics.Accuracy()
n_batches = 10
for i in range(n_batches):
    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    # metric on current batch
    acc = metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}")

acc = metric.compute()
print(f"Accuracy on all data: {acc}")