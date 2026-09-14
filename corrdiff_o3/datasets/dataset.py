"""Standalone HDF5 dataset for the CorrDiff ozone model."""

from dataclasses import dataclass
import os
import re
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChannelMetadata:
    """Metadata used by CorrDiff output writers."""

    name: str
    level: str = ""
    auxiliary: bool = False


def _normalize(data: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return (data - minimum) / (maximum - minimum + 1e-6) * 2.0 - 1.0


def _parse_minmax_line(line: str) -> Tuple[str, float, float] | None:
    """Parse common ``mins/maxs`` normalization range formats."""
    parts = line.split()
    if len(parts) < 2:
        return None

    name = parts[0]
    values = re.findall(r"(?:mins?|maxs?)\s*=?\s*([-+0-9.eE]+)", line)
    if len(values) >= 2:
        return name, float(values[0]), float(values[1])

    minimum_match = re.search(r"mins?\s*=?\s*([-+0-9.eE]+)", line)
    maximum_match = re.search(r"maxs?\s*=?\s*([-+0-9.eE]+)", line)
    if minimum_match and maximum_match:
        return name, float(minimum_match.group(1)), float(maximum_match.group(1))

    return None


class WeatherDataset(Dataset):
    """Load paired ozone downscaling samples from one HDF5 data directory.

    The directory must contain ``input.h5``, ``output.h5``, and ``minmax.txt``.
    ``input.h5`` stores the ten low-resolution variables and ``output.h5``
    stores the four high-resolution variables listed below.
    """

    low_keys = (
        "O3",
        "blh",
        "tp",
        "t2m",
        "u10",
        "RH",
        "sp",
        "v10",
        "ssrd",
        "tcc",
    )
    high_keys = (
        "hO3",
        "frac_urban",
        "frac_industry_transport",
        "frac_forest",
    )

    def __init__(
        self,
        data_path: str,
        datatype: str = "h5",
        pipeline: dict | None = None,
        phase: str = "train",
        train_test_split: bool = True,
        data_len: int = -1,
        min_max: str = "minmax.txt",
        in_channels: Sequence[int] | None = None,
        out_channels: Sequence[int] | None = None,
    ) -> None:
        del train_test_split
        if datatype.lower() not in {"h5", "hdf5"}:
            raise ValueError("The ozone dataset only supports HDF5 input.")
        if phase not in {"train", "val", "test"}:
            raise ValueError("phase must be one of: train, val, test")
        if data_len == 0 or data_len < -1:
            raise ValueError("data_len must be -1 or a positive integer")

        self.dataroot = os.fspath(data_path)
        self.phase = phase
        self.normal_type = (pipeline or {}).get("normal_type", "min_max")
        if self.normal_type != "min_max":
            raise NotImplementedError("Only min_max normalization is supported.")

        self.in_channel_indices = tuple(
            range(len(self.low_keys)) if in_channels is None else in_channels
        )
        self.out_channel_indices = tuple(
            range(len(self.high_keys)) if out_channels is None else out_channels
        )
        self._validate_channel_indices(
            self.in_channel_indices, len(self.low_keys), "input"
        )
        self._validate_channel_indices(
            self.out_channel_indices, len(self.high_keys), "output"
        )

        self.input_metadata = [
            ChannelMetadata(self.low_keys[index]) for index in self.in_channel_indices
        ]
        self.output_metadata = [
            ChannelMetadata(self.high_keys[index]) for index in self.out_channel_indices
        ]
        self.in_channels = list(self.input_metadata)
        self.out_channels = list(self.output_metadata)

        self.low_file = os.path.join(self.dataroot, "input.h5")
        self.high_file = os.path.join(self.dataroot, "output.h5")
        self.minmax = self._read_minmax(os.path.join(self.dataroot, min_max))
        self._check_required_files()

        self.lr_shape = (17, 22)
        self.hr_shape = (144, 192)
        with h5py.File(self.high_file, "r") as file:
            self.length = len(file[self.high_keys[0]])

        self.length = self.length if data_len < 0 else min(self.length, data_len)
        self._low_handle = None
        self._high_handle = None

    @staticmethod
    def _validate_channel_indices(
        indices: Sequence[int], channel_count: int, label: str
    ) -> None:
        if any(index < 0 or index >= channel_count for index in indices):
            raise ValueError(f"Invalid {label} channel index: {indices}")

    def _check_required_files(self) -> None:
        for path in (self.low_file, self.high_file):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        for key in self.low_keys:
            if key not in self.minmax:
                raise KeyError(f"Missing normalization range for {key}")
        for key in self.high_keys:
            if key not in self.minmax:
                raise KeyError(f"Missing normalization range for {key}")

    def _read_minmax(self, path: str) -> Dict[str, Tuple[float, float]]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        ranges: Dict[str, Tuple[float, float]] = {}
        with open(path, encoding="utf-8") as file:
            for line in file:
                parsed = _parse_minmax_line(line.strip())
                if parsed is not None:
                    name, minimum, maximum = parsed
                    ranges[name] = (minimum, maximum)
        return ranges

    def _ensure_open(self) -> None:
        if self._low_handle is None:
            self._low_handle = h5py.File(self.low_file, "r")
        if self._high_handle is None:
            self._high_handle = h5py.File(self.high_file, "r")

    def image_shape(self) -> Tuple[int, int]:
        return self.hr_shape

    def input_channels(self) -> List[ChannelMetadata]:
        return self.in_channels

    def output_channels(self) -> List[ChannelMetadata]:
        return self.out_channels

    def normalize_input(self, data: np.ndarray) -> np.ndarray:
        return self._normalize_channels(data, self.in_channel_indices, self.low_keys)

    def normalize_output(self, data: np.ndarray) -> np.ndarray:
        return self._normalize_channels(data, self.out_channel_indices, self.high_keys)

    def denormalize_input(self, data: np.ndarray) -> np.ndarray:
        return self._denormalize_channels(data, self.in_channel_indices, self.low_keys)

    def denormalize_output(self, data: np.ndarray) -> np.ndarray:
        return self._denormalize_channels(data, self.out_channel_indices, self.high_keys)

    def _normalize_channels(
        self, data: np.ndarray, indices: Sequence[int], keys: Sequence[str]
    ) -> np.ndarray:
        result = np.asarray(data, dtype=np.float32).copy()
        for channel, index in enumerate(indices):
            minimum, maximum = self.minmax[keys[index]]
            result[..., channel, :, :] = _normalize(
                result[..., channel, :, :], minimum, maximum
            )
        return result

    def _denormalize_channels(
        self, data: np.ndarray, indices: Sequence[int], keys: Sequence[str]
    ) -> np.ndarray:
        result = np.asarray(data, dtype=np.float32).copy()
        for channel, index in enumerate(indices):
            minimum, maximum = self.minmax[keys[index]]
            result[..., channel, :, :] = (result[..., channel, :, :] + 1.0) * 0.5 * (
                maximum - minimum + 1e-6
            ) + minimum
        return result

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        self._ensure_open()

        low = np.stack(
            [
                _normalize(
                    np.asarray(self._low_handle[self.low_keys[channel]][index, :17, :22]),
                    *self.minmax[self.low_keys[channel]],
                )
                for channel in self.in_channel_indices
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        high = np.stack(
            [
                _normalize(
                    np.asarray(
                        self._high_handle[self.high_keys[channel]][index, :144, :192]
                    ),
                    *self.minmax[self.high_keys[channel]],
                )
                for channel in self.out_channel_indices
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        low_tensor = torch.from_numpy(low).unsqueeze(0)
        high_tensor = torch.from_numpy(high).unsqueeze(0)
        low_tensor = F.interpolate(
            low_tensor, size=self.hr_shape, mode="bilinear", align_corners=False
        ).squeeze(0)
        high_tensor = F.interpolate(
            high_tensor, size=self.hr_shape, mode="bilinear", align_corners=False
        ).squeeze(0)
        return high_tensor.float(), low_tensor.float()

    def __del__(self) -> None:
        for handle in (self._low_handle, self._high_handle):
            if handle is not None:
                handle.close()
