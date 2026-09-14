from typing import Iterable, Tuple, Union
import copy
import importlib.util
from pathlib import Path

import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.generative import InfiniteSampler


known_datasets = {}


def register_dataset(dataset_spec: str) -> None:
    if dataset_spec in known_datasets:
        return

    try:
        file_path, class_name = dataset_spec.split("::")
    except ValueError:
        raise ValueError(
            "Invalid dataset specification. Expected format: "
            "'path_to_file.py::dataset_class'"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Dataset file not found: {file_path}")
    if file_path.suffix != ".py":
        raise ValueError(f"Dataset file must be a Python file: {file_path}")

    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        dataset_class = getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Could not find dataset class '{class_name}' in {file_path}")

    known_datasets[dataset_spec] = dataset_class


def init_train_valid_datasets_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    validation_dataset_cfg: Union[dict, None] = None,
    validation: bool = True,
) -> Tuple[object, Iterable, Union[object, None], Union[Iterable, None]]:
    config = copy.deepcopy(dataset_cfg)
    dataset, dataset_iter = init_dataset_from_config(
        config, dataloader_cfg, batch_size=batch_size, seed=seed
    )

    if validation:
        valid_dataset_cfg = copy.deepcopy(config)
        if validation_dataset_cfg:
            valid_dataset_cfg.update(validation_dataset_cfg)
        valid_dataset, valid_dataset_iter = init_dataset_from_config(
            valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed
        )
    else:
        valid_dataset = valid_dataset_iter = None

    return dataset, dataset_iter, valid_dataset, valid_dataset_iter


def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
) -> Tuple[object, Iterable]:
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type", "cwb")
    if "validation" in dataset_cfg:
        del dataset_cfg["validation"]

    dataset_init_func = known_datasets[dataset_type]
    dataset_obj = dataset_init_func(**dataset_cfg)

    if dataloader_cfg is None:
        dataloader_cfg = {}

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
            worker_init_fn=None,
            **dataloader_cfg,
        )
    )

    return dataset_obj, dataset_iterator
