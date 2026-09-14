# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset loading utilities for the custom diffusion dataset."""

from pathlib import Path
import copy
import importlib.util
from typing import Any, Iterable, Tuple, Union

import torch
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.generative import InfiniteSampler

from .base import DownscalingDataset


# Custom datasets are loaded from ``path/to/file.py::ClassName`` at runtime.
known_datasets: dict[str, type[DownscalingDataset]] = {}


def register_dataset(dataset_spec: str) -> None:
    """Load and cache a custom dataset class from a file specification."""
    if dataset_spec in known_datasets:
        return

    try:
        file_name, class_name = dataset_spec.split("::", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            "Invalid dataset specification. Expected 'path/to/dataset.py::ClassName'."
        ) from exc

    file_path = Path(file_name).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    if file_path.suffix != ".py":
        raise ValueError(f"Dataset file must be a Python file: {file_path}")

    module_name = f"_corrdiff_dataset_{file_path.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, file_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load dataset module: {file_path}")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    try:
        dataset_class = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Could not find dataset class '{class_name}' in {file_path}"
        ) from exc

    if not isinstance(dataset_class, type):
        raise TypeError(f"Dataset object '{class_name}' must be a class")

    known_datasets[dataset_spec] = dataset_class


def init_train_valid_datasets_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    validation_dataset_cfg: Union[dict, None] = None,
    validation: bool = True,
) -> Tuple[
    DownscalingDataset,
    Iterable,
    Union[DownscalingDataset, None],
    Union[Iterable, None],
]:
    """Create the training dataset and, optionally, a validation dataset."""
    config = copy.deepcopy(dataset_cfg)
    dataset, dataset_iter = init_dataset_from_config(
        config, dataloader_cfg, batch_size=batch_size, seed=seed
    )

    if not validation:
        return dataset, dataset_iter, None, None

    valid_dataset_cfg = copy.deepcopy(config)
    if validation_dataset_cfg:
        valid_dataset_cfg.update(validation_dataset_cfg)
    valid_dataset, valid_dataset_iter = init_dataset_from_config(
        valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed
    )
    return dataset, dataset_iter, valid_dataset, valid_dataset_iter


def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
) -> Tuple[DownscalingDataset, Iterable]:
    """Create a custom dataset and its distributed infinite DataLoader."""
    config: dict[str, Any] = copy.deepcopy(dataset_cfg)
    dataset_type = config.pop("type")
    register_dataset(dataset_type)

    dataset_class = known_datasets[dataset_type]
    dataset_obj = dataset_class(**config)

    loader_config = {} if dataloader_cfg is None else dict(dataloader_cfg)
    dist = DistributedManager()
    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_size,
            **loader_config,
        )
    )
    return dataset_obj, dataset_iterator
