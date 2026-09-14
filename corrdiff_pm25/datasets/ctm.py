from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import torch
import torch.nn.functional as F


def standard_norm(data: torch.Tensor, min, max):
    eps = 1e-6
    data = (data - min) / (max - min + eps)
    return data * 2 - 1


def find_range_index(ranges, i):
    for index in range(1, len(ranges)):
        if ranges[index - 1] <= i < ranges[index]:
            return index - 1, i - ranges[index - 1]
    return None


class WeatherDataset(Dataset):
    def __init__(
        self,
        data_path,
        datatype,
        pipeline,
        phase="train",
        train_test_split=True,
        data_len=-1,
        min_max="minmax.txt",
        in_channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        out_channels=[0, 1, 2, 3],
    ) -> None:
        super().__init__()
        if datatype != "h5":
            raise ValueError(
                "only support h5 file. nc format file does not support multi process reading"
            )
        if phase not in ("train", "val", "test"):
            raise ValueError(f"not supported phase: {phase}")

        self.dataroot = data_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.phase = phase
        self.low_keys = [
            "PM25",
            "blh",
            "tp",
            "t2m",
            "u10",
            "RH",
            "sp",
            "v10",
            "ssrd",
            "tcc",
        ]
        self.high_keys = [
            "hPM25",
            "frac_urban",
            "frac_industry_transport",
            "frac_forest",
        ]

        self.minmax = {}
        with open(os.path.join(self.dataroot, min_max)) as file:
            for line in file:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                varname = tokens[0]
                if varname in self.low_keys + self.high_keys:
                    min_value = line[line.find("mins") + 4 : line.find("mine")]
                    max_value = line[line.find("maxs") + 4 : line.find("maxe")]
                    self.minmax[varname] = (float(min_value), float(max_value))

        self.low_u_files = ["input.h5"]
        self.high_u_files = ["output.h5"]
        self.time_useful = [[] for _ in self.high_u_files]
        self.time_accum = [0]
        self.high_u_length = 0

        for index, filename in enumerate(self.high_u_files):
            with h5py.File(os.path.join(self.dataroot, filename), "r") as file:
                time_all = list(range(len(file[self.high_keys[0]])))
                self.time_useful[index] = time_all
                self.high_u_length += len(file[self.high_keys[0]])
                self.time_accum.append(self.high_u_length)

        self.low_u = [None for _ in self.low_u_files]
        self.high_u = [None for _ in self.high_u_files]
        self.length = self.high_u_length - 2 if phase == "train" else 2
        self.lr_shape = (17, 22)
        self.hr_shape = (144, 192)
        self.data_len = data_len
        self.step = 1
        self.normal_type = pipeline.get("normal_type", "min_max")

    def image_shape(self):
        return self.hr_shape

    def input_channels(self):
        return self.in_channels

    def output_channels(self):
        return self.out_channels

    def __len__(self):
        if self.phase == "test":
            return self.length // self.step
        if self.data_len < 0:
            return self.length
        return self.data_len

    def __getitem__(self, i):
        if self.phase in ("val", "test"):
            i += len(self.time_useful[-1]) - 2

        year_idx, diff = find_range_index(self.time_accum, i)
        time_idx = self.time_useful[year_idx][diff]

        if self.low_u[year_idx] is None:
            self.low_u[year_idx] = h5py.File(
                os.path.join(self.dataroot, self.low_u_files[year_idx]), "r"
            )
        if self.high_u[year_idx] is None:
            self.high_u[year_idx] = h5py.File(
                os.path.join(self.dataroot, self.high_u_files[year_idx]), "r"
            )

        low_data = []
        for key in self.low_keys:
            low = standard_norm(
                self.low_u[year_idx][key][time_idx : time_idx + 1, 0:17, 0:22],
                min=self.minmax[key][0],
                max=self.minmax[key][1],
            )
            low_data.append(low)

        high_data = []
        for key in self.high_keys:
            high = standard_norm(
                self.high_u[year_idx][key][time_idx : time_idx + 1, 0:144, 0:192],
                min=self.minmax[key][0],
                max=self.minmax[key][1],
            )
            high_data.append(high)

        low_tensor = torch.from_numpy(np.concatenate(low_data, axis=0))
        high_tensor = torch.from_numpy(np.concatenate(high_data, axis=0))
        input_tensor = F.interpolate(
            low_tensor.unsqueeze(0),
            size=self.hr_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).float()
        target_tensor = F.interpolate(
            high_tensor.unsqueeze(0),
            size=self.hr_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).float()

        if self.normal_type != "min_max":
            raise NotImplementedError

        return target_tensor, input_tensor
