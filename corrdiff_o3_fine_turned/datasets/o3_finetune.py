"""
O3 Finetuning Dataset — mirrors cma.py::WeatherDataset exactly.

Key differences from WeatherDataset:
- Uses O3/hO3 channel keys (not PM25/hPM25)
- Uses minmax_native.txt (2018-only, no key remapping)
- Train/val split: 2018 first 7944 steps train, last 744 steps val
- No clamp after standard_norm (confirmed: codebase has none)

Compatible with train.py, PhysicsNeMo loss functions, and generate scripts.
"""

from torch.utils.data import Dataset
import numpy as np
import os
import h5py

import torch
import torch.nn.functional as F


def standard_norm(data: torch.Tensor, vmin, vmax):
    eps = 1e-6
    data = (data - vmin) / (vmax - vmin + eps)
    data = data * 2 - 1
    return data


def find_range_index(ranges, i):
    for index in range(1, len(ranges)):
        if ranges[index - 1] <= i < ranges[index]:
            return index - 1, i - ranges[index - 1]
    return None


class O3FinetuneDataset(Dataset):
    def __init__(self, data_path, datatype, pipeline,
                 phase='train', data_len=-1, min_max='minmax_native.txt'):
        super().__init__()
        assert datatype == "h5", "only support h5 file"

        self.dataroot = data_path
        self.min_max_file = min_max

        # Channel keys — strictly aligned with cma.py order (ssrd before tcc)
        self.low_keys = ['O3', 'blh', 'tp', 't2m', 'u10', 'RH', 'sp', 'v10', 'ssrd', 'tcc']
        self.high_keys = ['hO3', 'frac_urban', 'frac_industry_transport', 'frac_forest']

        # Load minmax — key names match channel keys directly, no remapping
        self.minmax = {}
        with open(os.path.join(self.dataroot, min_max)) as f:
            for l in f.readlines():
                tokens = l.strip().split()
                if len(tokens) < 2:
                    continue
                varname = tokens[0]
                if varname in self.low_keys + self.high_keys:
                    min_ = l[l.find('mins') + 4: l.find('mine')]
                    max_ = l[l.find('maxs') + 4: l.find('maxe')]
                    self.minmax[varname] = (float(min_), float(max_))

        # File lists (single-year 2018)
        self.low_u_files = ['input.h5']
        self.high_u_files = ['output.h5']

        # Scan time indices
        self.time_useful = [[] for _ in self.high_u_files]
        self.time_accum = [0]
        self.high_u_length = 0

        for i, f in enumerate(self.high_u_files):
            with h5py.File(os.path.join(self.dataroot, f), 'r') as file:
                time_all = list(range(len(file[self.high_keys[0]])))
                self.time_useful[i] = time_all
                self.high_u_length += len(time_all)
                self.time_accum.append(self.high_u_length)

        # Delayed h5 handles
        self.low_u = [None for _ in self.low_u_files]
        self.high_u = [None for _ in self.high_u_files]

        total_steps = self.high_u_length  # 8688
        val_len = 744                       # ~31 days in December 2018

        if phase == 'train':
            self.length = total_steps - val_len           # 7944
            self._index_offset = 0
        elif phase == 'val' or phase == 'test':
            self.length = val_len                          # 744
            self._index_offset = total_steps - val_len     # start at 7944
        elif phase == 'inference':
            self.length = total_steps                      # all data
            self._index_offset = 0
        else:
            raise ValueError(f'not supported phase: {phase}')

        self.lr_shape = (17, 22)
        self.hr_shape = (144, 192)
        self.data_len = data_len
        self.phase = phase
        self.step = 1
        self.normal_type = pipeline.get('normal_type', 'min_max')

    def image_shape(self):
        return self.hr_shape

    def input_channels(self):
        return list(range(len(self.low_keys)))

    def output_channels(self):
        return list(range(len(self.high_keys)))

    def __len__(self):
        if self.data_len < 0:
            return self.length
        return self.data_len

    def __getitem__(self, i):
        # Phase-dependent offset
        i = i + self._index_offset

        year_idx, diff = find_range_index(self.time_accum, i)
        i_missing = self.time_useful[year_idx][diff]

        # Delayed open
        if self.low_u[year_idx] is None:
            self.low_u[year_idx] = h5py.File(
                os.path.join(self.dataroot, self.low_u_files[year_idx]), 'r')
        if self.high_u[year_idx] is None:
            self.high_u[year_idx] = h5py.File(
                os.path.join(self.dataroot, self.high_u_files[year_idx]), 'r')

        low_data_array_u = []
        for k in self.low_keys:
            low = standard_norm(
                self.low_u[year_idx][k][i_missing:i_missing + 1, 0:17, 0:22],
                self.minmax[k][0], self.minmax[k][1],
            )
            low_data_array_u.append(low)

        high_data_array_u = []
        for k in self.high_keys:
            high = standard_norm(
                self.high_u[year_idx][k][i_missing:i_missing + 1, 0:144, 0:192],
                self.minmax[k][0], self.minmax[k][1],
            )
            high_data_array_u.append(high)

        low_data_array_u = np.concatenate(low_data_array_u, axis=0)
        high_data_array_u = np.concatenate(high_data_array_u, axis=0)

        lr_tensor_u = torch.from_numpy(low_data_array_u)
        hr_tensor_u = torch.from_numpy(high_data_array_u)

        sr_tensor_u = F.interpolate(
            lr_tensor_u.unsqueeze(0), size=self.hr_shape,
            mode='bilinear', align_corners=False,
        ).squeeze(0).float()

        hr_tensor_u = F.interpolate(
            hr_tensor_u.unsqueeze(0), size=self.hr_shape,
            mode='bilinear', align_corners=False,
        ).squeeze(0).float()

        if self.normal_type != "min_max":
            raise NotImplementedError

        return hr_tensor_u, sr_tensor_u
