import math
import random
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np


class SingleviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)


class Incomplete_MultiviewDataset(Dataset):
    def __init__(self, data_list, mask_matrix, labels, num_views):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels
        self.mask_list = np.split(mask_matrix, num_views, axis=1)

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, index):
        data = [torch.tensor(self.data_list[v][index], dtype=torch.float32) for v in range(self.num_views)]
        mask = [torch.tensor(self.mask_list[v][index], dtype=torch.float32, requires_grad=False) for v in range(self.num_views)]
        return data, mask


def get_mask(num_views, data_size, missing_rate):
    assert num_views >= 2
    miss_sample_num = math.floor(data_size * missing_rate)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, num_views])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(num_views)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if 0 < np.sum(rand_v) < num_views:
                break
        mask[miss_ind[j]] = rand_v
    return mask


def load_multiview_data(args):
    data_path = args.dataset_dir_base + args.dataset_name + '.npz'
    data = np.load(data_path)
    num_views = int(data['n_views'])
    data_list = [data[f'view_{v}'].astype(np.float32) for v in range(num_views)]
    labels = data['labels']
    dims = [dv.shape[1] for dv in data_list]
    data_size = labels.shape[0]
    class_num = len(np.unique(labels))
    if np.max(labels) == class_num:
        labels = labels - 1
    args.multiview_dims = dims
    args.num_views = num_views
    args.class_num = class_num
    args.data_size = data_size
    return data_list, labels


def pixel_normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def build_dataset(args):
    data_list, labels = load_multiview_data(args)

    if args.dataset_name == 'Caltech7-5V':
        data_list = [pixel_normalize(dv) for dv in data_list]
    elif args.dataset_name == 'Scene-15':
        data_list = [StandardScaler().fit_transform(dv) for dv in data_list]
    else:
        pass

    mask = get_mask(args.num_views, args.data_size, args.missing_rate)
    data_list = [data_list[v] * mask[:, v:v + 1] for v in range(args.num_views)]
    incomplete_multiview_dataset = Incomplete_MultiviewDataset(data_list, mask, labels, args.num_views)

    com_idx = np.sum(mask, axis=1) == args.num_views
    complete_multiview_data = [sv_d[com_idx] for sv_d in data_list]

    exist_singleview_datasets = [SingleviewDataset(data_list[v][mask[:, v] == 1]) for v in range(args.num_views)]

    return complete_multiview_data, incomplete_multiview_dataset, exist_singleview_datasets
