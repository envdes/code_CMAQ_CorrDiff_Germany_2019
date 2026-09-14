from torch.utils.data import Dataset
import numpy as np
import os
import h5py

import torch
import torch.nn.functional as F

def standard_norm(data: torch.Tensor, min, max):
    data = (data - (min)) / (max - (min)) # scale to [0, 1]
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
        phase: train, val, test, or inference.
        """
        super().__init__()
        assert datatype == "h5", "only support h5 file. nc format file does not support multi process reading"
        self.dataroot = data_path
        self.in_channels, self.out_channels = in_channels, out_channels
        self.phase = phase # 保存 phase

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
        
        # Select the files required by each phase.
        if self.phase in ['train', 'val', 'test']:
            self.low_u_files = ['input.h5']
            self.high_u_files = ['output.h5']
        elif self.phase == 'inference':
            # 在推理模式下，我们只需要低分辨率输入文件
            self.low_u_files = ['input.h5']
            self.high_u_files = [] # 高分辨率文件列表为空
        else:
            raise ValueError(f'not supported phase: {self.phase}')

        self.time_useful = [[] for _ in self.low_u_files] # 使用 low_u_files 初始化
        self.time_accum = [0]
        self.data_total_length = 0

        # Determine dataset length from the files available in each phase.
        if self.phase == 'inference':
            # 推理模式下，长度由输入文件决定
            for i, f in enumerate(self.low_u_files):
                with h5py.File(os.path.join(self.dataroot, f), 'r') as file:
                    time_all = list(range(len(file[self.low_keys[0]])))
                    self.time_useful[i] = time_all
                    self.data_total_length += len(time_all)
                    self.time_accum.append(self.data_total_length)
            self.length = self.data_total_length
        else:
            # 训练、验证、测试模式下，长度由输出文件决定 (保持原逻辑)
            for i, f in enumerate(self.high_u_files):
                with h5py.File(os.path.join(self.dataroot, f), 'r') as file:
                    time_all = list(range(len(file[self.high_keys[0]])))
                    self.time_useful[i] = time_all
                    self.data_total_length += len(time_all)
                    self.time_accum.append(self.data_total_length)
            
            if phase == 'train':
                self.length = self.data_total_length - 2 # 最后2个时刻用于测试
            if phase == 'val' or phase == 'test':
                self.length = 2 # 最后2个时刻

        # get file handle
        self.low_u = [None for _ in self.low_u_files]
        self.high_u = [None for _ in self.high_u_files]
        print("daizx")
        print("low_path_u: ", self.low_u_files)
        print("high_path_u: ", self.high_u_files)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr_shape = (17, 22)
        self.hr_shape = (144, 192)

        self.data_len = data_len
        self.step = 1
        self.normal_type = pipeline.get('normal_type', 'min_max')

    def image_shape(self):
        return self.hr_shape

    def input_channels(self):
        return self.in_channels

    def output_channels(self):
        return self.out_channels

    def __len__(self):
        if self.data_len > 0 and self.phase != 'inference':
            return self.data_len
        return self.length // self.step

    def __getitem__(self, i):
        # Validation and test samples are taken from the end of the dataset.
        if self.phase == 'val' or self.phase == 'test':
            # 验证/测试时，从数据末尾开始取
            i = i + self.data_total_length - self.length
        
        # 对于 'train' 和 'inference'，i 直接使用

        year_idx, diff = find_range_index(self.time_accum, i)
        
        # Inference only opens the low-resolution input file.
        # 在 'inference' 模式下, self.time_useful 只有一个元素, year_idx 总是 0
        i_missing = self.time_useful[year_idx][diff]

        if self.low_u[year_idx] is None:
            self.low_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.low_u_files[year_idx]), 'r')
        
        # 只有在非推理模式下才需要打开高分辨率文件
        if self.phase != 'inference' and self.high_u[year_idx] is None:
            self.high_u[year_idx] = h5py.File(os.path.join(self.dataroot, self.high_u_files[year_idx]), 'r')

        low_data_array_u = []
        for k in self.low_keys:
            low = standard_norm(self.low_u[year_idx][k][i_missing:i_missing+1, 0:17, 0:22], min=self.minmax[k][0], max=self.minmax[k][1])
            low_data_array_u.append(low)
        
        low_data_array_u = np.concatenate(low_data_array_u, axis=0)
        lr_tensor_u = torch.from_numpy(low_data_array_u)
        sr_tensor_u = F.interpolate(lr_tensor_u.unsqueeze(0), size=self.hr_shape, mode='bilinear', align_corners=False).squeeze(0).float()

        # Inference has no high-resolution target, so return a shape-compatible placeholder.
        if self.phase == 'inference':
            # 在推理模式下，我们没有真实的高分数据，所以创建一个全零的占位符
            # 它的形状和上采样后的低分数据完全一致
            high_data_array_u = np.zeros_like(sr_tensor_u.numpy())
        else:
            # 在其他模式下，正常从文件读取高分数据
            high_data_array_u = []
            for k in self.high_keys:
                high = standard_norm(self.high_u[year_idx][k][i_missing:i_missing+1, 0:144, 0:192], min=self.minmax[k][0], max=self.minmax[k][1])
                high_data_array_u.append(high)
            high_data_array_u = np.concatenate(high_data_array_u, axis=0)

        hr_tensor_u = torch.from_numpy(high_data_array_u)
        hr_tensor_u = F.interpolate(hr_tensor_u.unsqueeze(0), size=self.hr_shape, mode='bilinear', align_corners=False).squeeze(0).float()

        if self.normal_type != "min_max":
            raise NotImplementedError

        sr_tensor = sr_tensor_u
        hr_tensor = hr_tensor_u

        return hr_tensor, sr_tensor