"""
O3 Inference Dataset — reads O3/hO3 from h5 files but normalizes
using the training minmax (PM25/hPM25 ranges).

Config parameters:
    data_path:   directory containing input.h5 / output.h5 (O3 data)
    minmax_path: path to the training minmax.txt (with PM25/hPM25 entries)
    datatype:    must be "h5"
    pipeline:    dict with normal_type (e.g. 'min_max')
    phase:       'test' (default)
"""

import os
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def standard_norm(data: torch.Tensor, vmin: float, vmax: float):
    eps = 1e-6
    data = (data - vmin) / (vmax - vmin + eps)
    data = data * 2 - 1
    return data


def find_range_index(ranges, i):
    for index in range(1, len(ranges)):
        if ranges[index - 1] <= i < ranges[index]:
            return index - 1, i - ranges[index - 1]
    return None


class O3InferenceDataset(Dataset):
    """Dataset for O3 inference using training minmax.

    H5 variable names are O3/O3h, but normalization uses PM25/hPM25 ranges
    from the training minmax file.
    """

    def __init__(self, data_path, datatype, pipeline,
                 minmax_path=None,
                 phase='test', data_len=-1):
        super().__init__()
        assert datatype == "h5", "only support h5 file"
        self.dataroot = data_path
        self.minmax_path = minmax_path  # training minmax path (PM25 ranges)

        # H5 variable names (O3 data)
        self.low_keys = ['O3', 'blh', 'tp', 't2m', 'u10', 'RH', 'sp', 'v10', 'ssrd', 'tcc']
        self.high_keys = ['hO3', 'frac_urban', 'frac_industry_transport', 'frac_forest']

        # Map h5 variable names → training minmax names.
        # The training minmax uses PM25/hPM25; the O3 h5 files use O3/hO3.
        self._minmax_key_map = {
            'O3': 'PM25', 'hO3': 'hPM25',
        }
        # Reverse mapping so we can store values under h5 keys.
        self._h5_to_mm = self._minmax_key_map
        self._mm_to_h5 = {v: k for k, v in self._minmax_key_map.items()}

        # Load minmax from the specified training minmax file.
        # Store under the H5 variable names so generate_debug.py can do
        # dataset.minmax['hO3'] and get the training hPM25 range.
        minmax_file = minmax_path if minmax_path else os.path.join(self.dataroot, 'minmax.txt')
        self.minmax = {}
        with open(minmax_file) as f:
            lines = f.readlines()
            for l in lines:
                tokens = l.strip().split()
                if len(tokens) < 2:
                    continue
                varname = tokens[0]
                if varname in self._all_needed_minmax_keys():
                    min_ = l[l.find('mins')+4: l.find('mine')]
                    max_ = l[l.find('maxs')+4: l.find('maxe')]
                    # Store under the H5-key name for uniform lookup
                    store_key = self._mm_to_h5.get(varname, varname)
                    self.minmax[store_key] = (float(min_), float(max_))

        # Validate that all needed h5 keys are present in minmax
        for h5_key in self.low_keys + self.high_keys:
            assert h5_key in self.minmax, f"Missing minmax key: {h5_key}"

        self.low_u_files = ['input.h5']
        self.high_u_files = ['output.h5']

        self.time_useful = [[] for _ in self.high_u_files]
        self.time_accum = [0]
        self.high_u_length = 0

        for i, f in enumerate(self.high_u_files):
            with h5py.File(os.path.join(self.dataroot, f), 'r') as file:
                time_all = list(range(len(file[self.high_keys[0]])))
                self.time_useful[i] = time_all
                self.high_u_length += len(time_all)
                self.time_accum.append(self.high_u_length)

        self.low_u = [None for _ in self.low_u_files]
        self.high_u = [None for _ in self.high_u_files]

        if phase == 'test':
            self.length = self.high_u_length
        else:
            self.length = self.high_u_length

        self.lr_shape = (17, 22)
        self.hr_shape = (144, 192)
        self.data_len = data_len
        self.phase = phase
        self.step = 1
        self.normal_type = pipeline.get('normal_type', 'min_max')

    def _all_needed_minmax_keys(self):
        keys = set()
        for h5_key in self.low_keys + self.high_keys:
            keys.add(self._minmax_key_map.get(h5_key, h5_key))
        return keys

    def _minmax_for(self, h5_key):
        """Get (vmin, vmax) for a given h5 variable."""
        return self.minmax[h5_key]

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
        year_idx, diff = find_range_index(self.time_accum, i)
        i_missing = self.time_useful[year_idx][diff]

        if self.low_u[year_idx] is None:
            self.low_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.low_u_files[year_idx]), 'r')
        if self.high_u[year_idx] is None:
            self.high_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.high_u_files[year_idx]), 'r')

        low_data_array_u, high_data_array_u = [], []
        for k in self.low_keys:
            vmin, vmax = self._minmax_for(k)
            low = standard_norm(
                self.low_u[year_idx][k][i_missing:i_missing+1, 0:17, 0:22],
                vmin, vmax,
            )
            low_data_array_u.append(low)

        for k in self.high_keys:
            vmin, vmax = self._minmax_for(k)
            high = standard_norm(
                self.high_u[year_idx][k][i_missing:i_missing+1, 0:144, 0:192],
                vmin, vmax,
            )
            high_data_array_u.append(high)

        low_data_array_u = np.concatenate(low_data_array_u, axis=0)
        high_data_array_u = np.concatenate(high_data_array_u, axis=0)

        lr_tensor_u = torch.from_numpy(low_data_array_u)
        hr_tensor_u = torch.from_numpy(high_data_array_u)
        sr_tensor_u = F.interpolate(lr_tensor_u.unsqueeze(0), size=self.hr_shape, mode='bilinear', align_corners=False).squeeze(0).float()
        hr_tensor_u = F.interpolate(hr_tensor_u.unsqueeze(0), size=self.hr_shape, mode='bilinear', align_corners=False).squeeze(0).float()

        if self.normal_type != "min_max":
            raise NotImplementedError

        return hr_tensor_u, sr_tensor_u
