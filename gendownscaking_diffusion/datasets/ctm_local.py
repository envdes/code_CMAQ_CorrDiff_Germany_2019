"""Local copy of WeatherDataset extended with generation-compatible methods.

Adds ChannelMetadata, denormalize_input/output, time(), longitude(), latitude() —
all needed by generate_helpers.save_images and generate.py.
"""

from io import BytesIO
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset
import random
import math
import numpy as np
import os
import h5py

import torch
import torchvision
import torch.nn.functional as F


@dataclass
class ChannelMetadata:
    """Metadata describing a data channel."""
    name: str
    level: str = ""
    auxiliary: bool = False


# Channel name lists — output is now single-channel hPM25 (log domain).
# 3 static frac fields moved from output to input conditioning.
INPUT_NAMES = ['PM25', 'blh', 'tp', 't2m', 'u10', 'RH', 'sp', 'v10', 'ssrd', 'tcc',
               'frac_urban', 'frac_industry_transport', 'frac_forest']   # 13
OUTPUT_NAMES = ['hPM25']                                                   # 1

# Channels that need log1p transform before min-max normalisation.
LOG_CHANNELS = {'PM25', 'hPM25'}


def standard_norm(data: torch.Tensor, min, max):
    eps = 1e-6
    data = (data - (min)) / (max - (min) + eps)
    data = data * 2 - 1
    return data


def standard_denorm(data: np.ndarray, min_val, max_val):
    """Inverse of standard_norm: maps [-1, 1] back to [min, max]."""
    eps = 1e-6
    data = (data + 1) / 2  # [0, 1]
    return data * (max_val - min_val + eps) + min_val


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

        self.low_keys = ['PM25', 'blh', 'tp', 't2m', 'u10', 'RH', 'sp', 'v10', 'ssrd', 'tcc']
        self.high_keys = ['hPM25', 'frac_urban', 'frac_industry_transport', 'frac_forest']

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
            self.low_u_files = ['input.h5']
            self.high_u_files = ['output.h5']
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
            self.length = self.high_u_length - 2
        if phase == 'val' or phase == 'test':
            self.length = 2

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

    # ------------------------------------------------------------------
    # Methods required by train.py (channel counts, image shape)
    # ------------------------------------------------------------------

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return self.hr_shape

    def input_channels(self):
        """Return ChannelMetadata list — 13 channels (10 met + 3 frac)."""
        return [ChannelMetadata(name=n) for n in INPUT_NAMES]

    def output_channels(self):
        """Return ChannelMetadata list — single hPM25 channel (log domain)."""
        return [ChannelMetadata(name=n) for n in OUTPUT_NAMES]

    # ------------------------------------------------------------------
    # Methods required by generate_helpers.save_images (denormalize)
    # ------------------------------------------------------------------

    def denormalize_input(self, x):
        """Denormalize [-1,1] → physical units for 13 input channels.

        Channels 0-9 are met vars (log1p-expm1 for PM25), channels 10-12 are frac fields.
        """
        x = np.asarray(x)
        out = np.zeros_like(x)
        for k, name in enumerate(INPUT_NAMES):
            mn, mx = self.minmax[name]
            if name in LOG_CHANNELS:
                lm, lM = math.log1p(mn), math.log1p(mx)
                out[:, k, :, :] = np.expm1((x[:, k, :, :] + 1) * 0.5 * (lM - lm + 1e-6) + lm)
            else:
                out[:, k, :, :] = standard_denorm(x[:, k, :, :], mn, mx)
        return out

    def denormalize_output(self, x):
        """Denormalize [-1,1] → physical µg/m³ for single hPM25 channel (was log domain)."""
        x = np.asarray(x)
        out = np.zeros_like(x)
        mn, mx = self.minmax['hPM25']
        lm, lM = math.log1p(mn), math.log1p(mx)
        out[:, 0, :, :] = np.expm1((x[:, 0, :, :] + 1) * 0.5 * (lM - lm + 1e-6) + lm)
        return out

    # ------------------------------------------------------------------
    # Methods required by generate_helpers.get_dataset_and_sampler
    # ------------------------------------------------------------------

    def time(self):
        """Return list of absolute time indices (0 … length-1)."""
        return list(range(self.length))

    def longitude(self):
        """Stub — generation needs this but the CMA dataset has no lon/lat grid."""
        return np.arange(self.hr_shape[1])

    def latitude(self):
        """Stub — generation needs this but the CMA dataset has no lon/lat grid."""
        return np.arange(self.hr_shape[0])

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

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

    def _norm_channel(self, x, name, mn, mx):
        """Normalise a raw tensor to [-1,1], optionally applying log1p first."""
        if name in LOG_CHANNELS:
            x = torch.log1p(x)
            mn, mx = math.log1p(mn), math.log1p(mx)
        return standard_norm(x, mn, mx)

    def __getitem__(self, i):
        if self.phase == 'val' or self.phase == 'test':
            i = i + len(self.time_useful[-1]) - 2
        year_idx, diff = find_range_index(self.time_accum, i)
        i_missing = self.time_useful[year_idx][diff]

        if self.low_u[year_idx] is None:
            self.low_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.low_u_files[year_idx]), 'r')
        if self.high_u[year_idx] is None:
            self.high_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.high_u_files[year_idx]), 'r')

        # --- meteorological conditions (10 channels, LR → upsample) ---
        met_names = ['PM25', 'blh', 'tp', 't2m', 'u10', 'RH', 'sp', 'v10', 'ssrd', 'tcc']
        met_tensors = []
        for name in met_names:
            raw = torch.from_numpy(
                self.low_u[year_idx][name][i_missing:i_missing + 1, 0:17, 0:22]
            ).float()
            normed = self._norm_channel(raw, name, *self.minmax[name])
            met_tensors.append(normed)
        met_cat = torch.cat(met_tensors, dim=0)  # (10, 17, 22)
        met_up = F.interpolate(met_cat.unsqueeze(0), size=self.hr_shape,
                               mode='bilinear', align_corners=False).squeeze(0)  # (10, 144, 192)

        # --- static frac fields (3 channels, native HR, → conditioning) ---
        frac_names = ['frac_urban', 'frac_industry_transport', 'frac_forest']
        frac_tensors = []
        for name in frac_names:
            raw = torch.from_numpy(
                self.high_u[year_idx][name][i_missing:i_missing + 1, 0:144, 0:192]
            ).float()
            normed = self._norm_channel(raw, name, *self.minmax[name])
            frac_tensors.append(normed)
        frac_cat = torch.cat(frac_tensors, dim=0)  # (3, 144, 192)

        # --- conditioning: 10 met + 3 frac = 13 channels ---
        sr_tensor = torch.cat([met_up, frac_cat], dim=0)  # (13, 144, 192)

        # --- target: hPM25 in log domain, 1 channel ---
        raw_hpm25 = torch.from_numpy(
            self.high_u[year_idx]['hPM25'][i_missing:i_missing + 1, 0:144, 0:192]
        ).float()
        hr_tensor = self._norm_channel(raw_hpm25, 'hPM25', *self.minmax['hPM25'])  # (1, 144, 192)

        if self.normal_type != "min_max":
            raise NotImplementedError

        return hr_tensor, sr_tensor
