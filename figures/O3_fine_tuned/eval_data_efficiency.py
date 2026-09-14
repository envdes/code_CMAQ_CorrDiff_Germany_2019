#!/usr/bin/env python3
"""Evaluate O3 data-efficiency experiments from saved regression checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from physicsnemo import Module

from corrdiff_o3_fine_turned.datasets.o3_finetune import O3FinetuneDataset

H_O3_VMIN = 4.586369261802559e-13
H_O3_VMAX = 121.23634338378906


def denorm(values):
    return (values + 1.0) / 2.0 * (H_O3_VMAX - H_O3_VMIN) + H_O3_VMIN


def experiments(checkpoint_root):
    return [
        ("full", "warm-start", "full", 7944),
        ("full", "retrain", "full", 7944),
        ("6mon", "warm-start", "6mon", 4344),
        ("6mon", "retrain", "6mon", 4344),
        ("3mon", "warm-start", "3mon", 2172),
        ("3mon", "retrain", "3mon", 2172),
        ("1mon", "warm-start", "1mon", 720),
        ("1mon", "retrain", "1mon", 720),
        ("2wk", "warm-start", "2wk", 336),
        ("2wk", "retrain", "2wk", 336),
    ], checkpoint_root


def checkpoint_step(path):
    try:
        return int(path.name.split(".")[-2])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Unexpected checkpoint name: {path.name}") from error


def latest_checkpoint(directory):
    files = list(directory.glob("checkpoints_regression/UNet.0.*.mdlus"))
    if not files:
        return None
    return max(files, key=checkpoint_step)


def evaluate_checkpoint(checkpoint, dataset, truth, channel_index, device):
    network = Module.from_checkpoint(str(checkpoint), {"amp_mode": False})
    network.eval().to(device).to(memory_format=torch.channels_last)

    squared_error = 0.0
    total_values = 0
    for index in range(len(dataset)):
        hr, lr = dataset[index]
        lr_batch = lr.unsqueeze(0).to(device).to(memory_format=torch.channels_last)
        hr_batch = hr.unsqueeze(0).to(device).to(memory_format=torch.channels_last)
        latent = torch.zeros(
            1,
            hr_batch.shape[1],
            *hr_batch.shape[2:],
            dtype=hr_batch.dtype,
            device=device,
        )
        with torch.inference_mode():
            prediction = network(x=latent, img_lr=lr_batch)

        predicted = denorm(prediction[0, channel_index].cpu().numpy())
        squared_error += np.square(predicted - truth[index]).sum()
        total_values += predicted.size

    del network
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(np.sqrt(squared_error / total_values))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate warm-start and retrained O3 models.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--min-max", default="minmax_native.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("data_efficiency_results.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dataset = O3FinetuneDataset(
        str(args.data),
        datatype="h5",
        pipeline={"normal_type": "min_max"},
        phase="inference",
        min_max=args.min_max,
    )

    truth = []
    for index in range(len(dataset)):
        hr, _ = dataset[index]
        truth.append(hr.numpy())
    truth = denorm(np.stack(truth))
    channel_index = dataset.high_keys.index("hO3")

    rows = []
    definitions, root = experiments(args.checkpoint_root)
    for subset, initialization, subset_name, sample_count in definitions:
        checkpoint_dir = root / f"checkpoints_data_eff_{'ft' if initialization == 'warm-start' else 'rt'}_{subset}"
        checkpoint = latest_checkpoint(checkpoint_dir)
        if checkpoint is None:
            continue

        rmse = evaluate_checkpoint(
            checkpoint,
            dataset,
            truth[:, channel_index],
            channel_index,
            device,
        )
        observed = truth[:, channel_index].ravel()
        ss_tot = np.sum((observed - observed.mean()) ** 2)
        squared_error = rmse**2 * observed.size
        r2 = 1.0 - squared_error / ss_tot if ss_tot > 0 else 0.0
        rows.append(
            {
                "init": initialization,
                "train_subset": subset_name,
                "n_samples": sample_count,
                "steps": checkpoint_step(checkpoint),
                "RMSE_2019": round(rmse, 4),
                "R2_2019": round(float(r2), 4),
            }
        )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
