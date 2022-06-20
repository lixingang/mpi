import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from collections import Counter
from scipy.ndimage import convolve1d
import torch


def _get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ["gaussian", "tria ng", "laplace"]
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        # base_kernel = [0.1] * half_ks + [1.0] + [0.1] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
            gaussian_filter1d(base_kernel, sigma=sigma)
        )
    elif kernel == "triang":
        kernel_window = triang(ks)
    else:

        def laplace(x):
            return np.exp(-abs(x) / sigma) / (2.0 * sigma)

        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1))
        )

    return kernel_window


def _get_bin_idx(x):
    # label = label.detach().cpu().numpy()
    return min(int(x * np.float32(100)), 90)


def get_lds_weights(labels):
    # preds, labels: [Ns,], "Ns" is the number of total samples
    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,]
    if not isinstance(labels, list):
        labels = torch.squeeze(labels)
    bin_index_per_label = [_get_bin_idx(label) for label in labels]
    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    # print("bin_index_per_label",len(bin_index_per_label),bin_index_per_label)
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = _get_lds_kernel_window(kernel="gaussian", ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(
        np.array(emp_label_dist), weights=lds_kernel_window, mode="constant"
    )

    # print(len(eff_label_dist),eff_label_dist)
    # print(bin_index_per_label)
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    # print(eff_num_per_label)
    weights = torch.tensor([np.float32(1 / x) for x in eff_num_per_label]).cuda()

    return weights
