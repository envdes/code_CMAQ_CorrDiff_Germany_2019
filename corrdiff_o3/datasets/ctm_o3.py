from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
import os
import h5py

import torch
import torchvision
import torch.nn.functional as F

def standard_norm(data: torch.Tensor, min, max):
    eps = 1e-6
    data = (data - (min)) / (max - (min) + eps)
    data = data * 2 - 1
    return data

def find_range_index(ranges, i):
    for index in range(1, len(ranges)):
        if ranges[index - 1] <= i < ranges[index]:
            return index - 1, i - ranges[index - 1]
    return None

class WeatherDataset(Dataset):
    def __init__(self, data_path, datatype, pipeline,
                 phase='train', train_test_split=True, data_len=-1, min_max='minmax.txt',
                 in_channels=[0,1,2,3,4,5,6,7,8,9], out_channels=[0,1,2,3]) -> None:
        """
        phase: train, val, or test.
        """
        super().__init__()
        assert datatype == "h5", "only support h5 file. nc format file does not support multi process reading"
        self.dataroot = data_path
        self.in_channels, self.out_channels = in_channels, out_channels

        self.low_keys = ['O3', 'blh', 'tp', 't2m', 'u10', 'RH', 'sp', 'v10', 'ssrd', 'tcc']
        self.high_keys = ['hO3', 'frac_urban', 'frac_industry_transport', 'frac_forest']

        self.minmax = {}
        with open(os.path.join(self.dataroot, min_max)) as f:
            lines = f.readlines()
            for l in lines:
                tokens = l.strip().split()
                if len(tokens) < 2:
                    continue
                varname = tokens[0]
                if varname in self.low_keys + self.high_keys:
                    min_ = l[l.find('mins')+4: l.find('mine')]
                    max_ = l[l.find('maxs')+4: l.find('maxe')]
                    self.minmax[varname] = (float(min_), float(max_))

        if phase == 'train':
            self.low_u_files = ['input.h5',
                           ]
            self.high_u_files = ['output.h5',
                           ]

        elif phase == 'val' or phase == 'test':
            self.low_u_files = ['input.h5']
            self.high_u_files = ['output.h5']
        else:
            raise ValueError(f'not supported phase: {self.phase}')

        self.time_useful = [[] for _ in self.high_u_files]
        self.time_accum = [0]

        self.high_u_length = 0
        for i, f in enumerate(self.high_u_files):
            with h5py.File(os.path.join(self.dataroot, f), 'r') as file:
                time_all = list(range(len(file[self.high_keys[0]])))
                self.time_useful[i] = time_all
                self.high_u_length += (len(file[self.high_keys[0]]))
                self.time_accum.append(self.high_u_length)

        self.low_u = [None for _ in self.low_u_files]
        self.high_u = [None for _ in self.high_u_files]
        print("daizx")

        if phase == 'train':
            self.length = self.high_u_length
        if phase == 'val' or phase == 'test':
            self.length = self.high_u_length

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr_shape = (17, 22)
        self.hr_shape = (144, 192)

        self.data_len = data_len

        self.phase = phase
        if phase == 'test':
            self.step = 1
        else:
            self.step = 1

        self.normal_type = pipeline.get('normal_type', 'min_max')

    def image_shape(self):
        return self.hr_shape

    def input_channels(self):
        return self.in_channels

    def output_channels(self):
        return self.out_channels

    def __len__(self):
        if self.phase == 'test':
            print("test phase. data len: self.time_idx_len // self.step = ", self.length // self.step)
            return self.length // self.step

        if self.data_len < 0:
            print("data len: ", self.length)
            return self.length
        else:
            print("data len: ", self.data_len)
            return self.data_len

    def __getitem__(self, i):
        year_idx, diff = find_range_index(self.time_accum, i)
        i_missing = self.time_useful[year_idx][diff]

        if self.low_u[year_idx] is None:
            self.low_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.low_u_files[year_idx]), 'r')
        if self.high_u[year_idx] is None:
            self.high_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.high_u_files[year_idx]), 'r')

        low_data_array_u, high_data_array_u = [], []

        for k in self.low_keys:
            low = standard_norm(self.low_u[year_idx][k][i_missing:i_missing+1, 0:17, 0:22], min=self.minmax[k][0], max=self.minmax[k][1])
            low_data_array_u.append(low)

        for k in self.high_keys:
            high = standard_norm(self.high_u[year_idx][k][i_missing:i_missing+1, 0:144, 0:192], min=self.minmax[k][0], max=self.minmax[k][1])
            high_data_array_u.append(high)

        low_data_array_u = np.concatenate(low_data_array_u, axis=0)
        high_data_array_u = np.concatenate(high_data_array_u, axis=0)

        lr_tensor_u = torch.from_numpy(low_data_array_u)
        hr_tensor_u = torch.from_numpy(high_data_array_u)
        sr_tensor_u = F.interpolate(lr_tensor_u.unsqueeze(0), size=self.hr_shape, mode='bilinear', align_corners=False).squeeze(0).float()
        hr_tensor_u = F.interpolate(hr_tensor_u.unsqueeze(0), size=self.hr_shape, mode='bilinear', align_corners=False).squeeze(0).float()

        if self.normal_type != "min_max":
            raise NotImplementedError

        sr_tensor = sr_tensor_u
        hr_tensor = hr_tensor_u

        return hr_tensor, sr_tensor
