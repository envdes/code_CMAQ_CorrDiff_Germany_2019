from torch.utils.data import Dataset
import numpy as np
import os
import h5py

import torch
import torch.nn.functional as F

def standard_norm(data: torch.Tensor, min, max):
    eps = 1e-6  # 设置一个极小值
    data = (data - (min)) / (max - (min) + eps)
#    data = (data - (min)) / (max - (min)) # scale to [0, 1]
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
#                 in_channels=['PM25', 'blh', 'tp', 't2m', 'u10', 'RH', 'sp', 'v10', 'ssrd', 'tcc'], out_channels=['hPM25', 'frac_urban', 'frac_industry_transport', 'frac_forest']) -> None:
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
                # 提取当前行的变量名（按空格分隔）
                tokens = l.strip().split()
                if len(tokens) < 2:
                    continue
                varname = tokens[0]
                if varname in self.low_keys + self.high_keys:
                    min_ = l[l.find('mins')+4: l.find('mine')]
                    max_ = l[l.find('maxs')+4: l.find('maxe')]
                    self.minmax[varname] = (float(min_), float(max_))

        if phase == 'train':
            #TODO
            self.low_u_files = ['input.h5',
                           ] # 25 km
            self.high_u_files = ['output.h5',
                           ] # 3 km

        elif phase == 'val' or phase == 'test':
            self.low_u_files = ['input.h5']  # 25 km
            self.high_u_files = ['output.h5'] # 3 km
        else:
            raise ValueError(f'not supported phase: {self.phase}')

#        # time accumulation
#        self.DEM = torch.FloatTensor(np.load(os.path.join(self.dataroot, 'crop_dem_solar_east_0.npy'))).unsqueeze(0)
#        #self.DEM = standard_norm(self.DEM, -156.0, 6524.0)
#        self.DEM = standard_norm(self.DEM, 0, 3576.0)
#
#        self.time_missing_dic = np.load(os.path.join(self.dataroot, 'time_missing_or_night_3_11.npy'), allow_pickle=True).item()
        self.time_useful = [[] for _ in self.high_u_files]
        self.time_accum = [0]

        self.high_u_length = 0
        for i, f in enumerate(self.high_u_files):
            with h5py.File(os.path.join(self.dataroot, f), 'r') as file:
                time_all = list(range(len(file[self.high_keys[0]])))
#                time_missing = self.time_missing_dic[f[f.find('/')+1:]]
#                self.time_useful[i] = [item for item in time_all if item not in time_missing]
                self.time_useful[i] = time_all
                self.high_u_length += (len(file[self.high_keys[0]]))
#                self.high_u_length += (len(file[self.high_keys[0]]) - len(self.time_missing_dic[f[f.find('/')+1:]]))
#                assert len(self.time_useful[i]) == (len(file[self.high_keys[0]]) - len(self.time_missing_dic[f[f.find('/')+1:]]))
                self.time_accum.append(self.high_u_length)
#        print(self.time_accum)
#        print('self.time_accum', self.time_accum)

        # get file handle
        self.low_u = [None for _ in self.low_u_files]
        self.high_u = [None for _ in self.high_u_files]
        print("daizx")
#        print(self.high_u)
        # print(self.low_u[0]["u10"].shape)
#        print("load data from:")
#        print("low_path_u: ", self.low_u_files)
#        print("high_path_u: ", self.high_u_files)
        # exit(1)
        if phase == 'train':
            self.length = self.high_u_length # 最后4个月测试
        if phase == 'val' or phase == 'test':
            self.length = self.high_u_length

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr_shape = (17, 22)
        self.hr_shape = (144, 192)

        ### config below
        self.data_len = data_len

        self.phase = phase
        if phase == 'test':
            self.step = 1  # 5
        else:
            self.step = 1

        self.normal_type = pipeline.get('normal_type', 'min_max')

    def image_shape(self):
        """Get the shape of the image (same for input and output)."""
        return self.hr_shape

    def input_channels(self):
        return self.in_channels

    def output_channels(self):
        return self.out_channels

#    def get_lr_idx(self, year_idx, hour_idx):
#        time_missing = self.time_missing[year_idx]
#        if len(time_missing) == 0:
#            return hour_idx
#        for i in range(len(time_missing)):
#            i = len(time_missing) - 1 - i
#            missing_hour = time_missing[i]
#            if hour_idx >= missing_hour:
#                return hour_idx + i + 1
#        return hour_idx

    def __len__(self):
        if self.phase == 'test':
            print("test phase. data len: self.time_idx_len // self.step = ", self.length // self.step)
            return self.length // self.step

        if self.data_len < 0: # train phase
            print("data len: ", self.length)
            return self.length
        else: # val phase
            print("data len: ", self.data_len)
            return self.data_len

    def __getitem__(self, i):
        # if self.phase == 'val' or self.phase == 'test': # 测试最后一年最后500个时刻
        #     i = i + len(self.time_useful[-1]) - 2
        year_idx, diff = find_range_index(self.time_accum, i)
        i_missing = self.time_useful[year_idx][diff]

        if self.low_u[year_idx] is None:
            self.low_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.low_u_files[year_idx]), 'r')
        if self.high_u[year_idx] is None:
            self.high_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.high_u_files[year_idx]), 'r')

        low_data_array_u, high_data_array_u = [], []
        # print(self.low_keys)
        # print(self.high_keys)
        for k in self.low_keys:
#            low = standard_norm(self.low_u[year_idx][k][i_missing:i_missing+1, :, :], min=self.minmax[k][0], max=self.minmax[k][1])
            low = standard_norm(self.low_u[year_idx][k][i_missing:i_missing+1, 0:17, 0:22], min=self.minmax[k][0], max=self.minmax[k][1])
#            print(f"Lowshape({k}): {low.shape}")
            low_data_array_u.append(low)

        for k in self.high_keys:
#            high = standard_norm(self.high_u[year_idx][k][i_missing:i_missing+1, :, :], min=self.minmax[k][0], max=self.minmax[k][1])
            high = standard_norm(self.high_u[year_idx][k][i_missing:i_missing+1, 0:144, 0:192], min=self.minmax[k][0], max=self.minmax[k][1])
#            print(f"Highshape({k}): {high.shape}")  # 打印高分辨率数据的形状
            high_data_array_u.append(high)

        low_data_array_u = np.concatenate(low_data_array_u, axis=0)
        high_data_array_u = np.concatenate(high_data_array_u, axis=0)

        # print(low_data_array_u.shape, high_data_array_u.shape)

        lr_tensor_u = torch.from_numpy(low_data_array_u)
        hr_tensor_u = torch.from_numpy(high_data_array_u)
        sr_tensor_u = F.interpolate(lr_tensor_u.unsqueeze(0), size=self.hr_shape, mode='bilinear', align_corners=False).squeeze(0).float()
        hr_tensor_u = F.interpolate(hr_tensor_u.unsqueeze(0), size=self.hr_shape, mode='bilinear', align_corners=False).squeeze(0).float()

        # normalization
        if self.normal_type != "min_max":
            raise NotImplementedError

        sr_tensor = sr_tensor_u
        hr_tensor = hr_tensor_u
#        sr_tensor = torch.concat([sr_tensor, self.DEM], dim=0)
#        exit(1)
    #    print("Input shape: ", sr_tensor.shape)
    #    print("Output shape: ", hr_tensor.shape)

        return hr_tensor, sr_tensor


