import torch
import numpy as np
import random


# class FeatureExtractor(torch.nn.Module):
#     def __init__(self, model, layers):
#         super().__init__()
#         self.model = model
#         self.layers = layers
#         self._features = {layer: torch.empty(0) for layer in layers}

#         for layer_id in layers:
#             layer = dict([*self.model.named_modules()])[layer_id]
#             layer.register_forward_hook(self.save_outputs_hook(layer_id))
#             layer.register_forward_hook(self.print_outputs_hook(layer_id))

#     def save_outputs_hook(self, layer_id: str):
#         def fn(_, __, output):
#             self._features[layer_id] = output
#         return fn

#     def print_outputs_hook(self, layer_id: str):
#         def fn(_, __, output):
#             print(f"{layer_id}: {output.shape}")
#         return fn

#     def forward(self, x):
#         output = self.model(x)
#         return output


class SaveOutput:
    def __init__(self):
        self.data = None

    def __call__(self, module, module_in, module_out):
        # print(module_out.clone().detach().shape)
        self.data = module_out.clone().detach().cpu()

        return None


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


class Meter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []

    def update(self, val):
        if isinstance(val, list):
            pass
        elif isinstance(val, np.ndarray):
            val = val.tolist()
        elif torch.is_tensor(val):
            if val.is_cuda:
                val = val.detach().cpu()
            # val = val.tolist()

        self.values.append(val)

    def avg(self):
        try:
            values_flatten = np.asarray(self.values).flatten()
            return np.average(values_flatten)
        except:
            print("An exception occurred, checkout the type of Meter")

    def cat(self, dim=0):
        assert len(self.values) != 0
        if isinstance(self.values[0], list):
            return list(np.concatenate(self.values).flatten())
        else:
            return torch.cat(self.values, dim=dim)
