CMAQ Air Pollution Downscaling
=============================

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Introduction](#introduction)
- [Scripts and Data](#scripts-and-data)
  - [Prerequisite](#prerequisite)
  - [Scripts](#scripts)
  - [Data](#data)
- [Acknowledgments](#acknowledgments)

<!-- /code_chunk_output -->

## Introduction

This repository accompanies the manuscript **"A Generative Downscaling Framework for Fine-scale Air Pollution Variability and Population Exposure"**.

The objectives of this project are:
- Downscale CMAQ-simulated **PM<sub>2.5</sub> from 36 km to 4 km** using CorrDiff, ERA5 meteorological variables, and CORINE land-cover fractions.
- Compare downscaling methods and assess **resolution-dependent population exposure**.
- Evaluate **PM<sub>2.5</sub>-to-O<sub>3</sub> transfer learning** through fine-tuning.

## Scripts and Data

### Prerequisite

- If you do not have the **conda** system

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

- Create and activate an environment for plotting

```bash
conda create -n cmaq-figures -c conda-forge python=3.11 numpy pandas matplotlib \
    xarray netcdf4 pyproj cartopy geopandas
conda activate cmaq-figures
```

### Scripts

| Tasks | Folders | Fig or Tab in paper | Fig or Tab in preprint |
| ----- | ------- | ------------------- | --------------------- |
| PM2.5 model development | [corrdiff_pm25](corrdiff_pm25/), [regdownscaling_unet](regdownscaling_unet/), [gendownscaking_diffusion](gendownscaking_diffusion/) | Tab 2 | |
| Tabular baseline models | [mldownscaling_flaml](mldownscaling_flaml/), [mldownscaling_randomforest](mldownscaling_randomforest/) | Tab 2 | |
| O3 training and fine-tuning | [corrdiff_o3](corrdiff_o3/), [corrdiff_o3_fine_turned](corrdiff_o3_fine_turned/) | Tab 3 | |
| Modeling-domain visualization | [figures/domain](figures/domain/) | Fig 1 (`domain.py`) | |
| PM2.5 performance, spatial maps, and distributions | [figures/Comparison of downscaling strategies](figures/Comparison%20of%20downscaling%20strategies/) | Fig 2 (`plot_figure2_monthly_performance.py`); Figs 3–4 (`plot_figure3_figure4_spatial_maps.py`); Fig 5 (`plot_figure5_kde_distribution.py`) | |
| Exceedance exposure and ensemble analysis | [figures/Resolution-dependent exceedance exposure](figures/Resolution-dependent%20exceedance%20exposure/) | Fig 6 (`plot_fig6_ensemble_robustness.py`); Fig 7 (`plot_fig7_resolution_recovery.py`); Fig 8 (`plot_fig8_population_redistribution.py`) | |
| O3 transfer-learning analysis | [figures/O3_fine_tuned](figures/O3_fine_tuned/) | Fig 9 (`plot_figure9_transfer_learning_efficiency.py`); Fig 10 (`plot_figure10_monthly_performance.py`) | |

To redraw the supplied CSV results for **Figs 2, 5, 9, and 10**, run the following from the repository root. The preparation step arranges the input files and standardizes column and model names without changing numerical values.

```bash
mkdir -p figures/data
cp "figures/Comparison of downscaling strategies/figure5_"*.csv figures/data/
cp figures/O3_fine_tuned/figure10_monthly_metrics.csv figures/data/

python - <<'PY'
import pandas as pd
source = "figures/Comparison of downscaling strategies/monthly_three_methods.csv"
df = pd.read_csv(source).rename(columns=str.lower)
df["model"] = df["model"].replace({"FLAML": "FLAML/XGBoost", "reg": "Reg"})
df.to_csv("figures/data/figure2_monthly_metrics.csv", index=False)
PY

python "figures/Comparison of downscaling strategies/plot_figure2_monthly_performance.py"
python "figures/Comparison of downscaling strategies/plot_figure5_kde_distribution.py"
python figures/O3_fine_tuned/plot_figure9_transfer_learning_efficiency.py \
    --input figures/O3_fine_tuned/data_efficiency.csv \
    --output-dir figures/O3_fine_tuned
python figures/O3_fine_tuned/plot_figure10_monthly_performance.py
```

PNG and PDF outputs are saved in the corresponding plotting folders.

### Data

The following files are provided in this repository. Additional study data will be released upon publication.

- `corrdiff_pm25`

  | Num | Folder | Comments | How to get it? |
  | --- | ------ | -------- | -------------- |
  | 1.1 | `pm25_corrdiff.npz` | CorrDiff daily PM2.5 output | Included |

- `figures/domain`

  | Num | Folder | Comments | How to get it? |
  | --- | ------ | -------- | -------------- |
  | 2.1 | `geo_em.d01.nc`; `namelist.wps` | Parent-domain geogrid data and nested-domain configuration | Included |

- `figures/Comparison of downscaling strategies`

  | Num | Folder | Comments | How to get it? |
  | --- | ------ | -------- | -------------- |
  | 3.1 | `monthly_three_methods.csv` | Monthly PM2.5 RMSE and R2 for six methods | Included |
  | 3.2 | `figure3_figure4_spatial_maps.npz` | January and July PM2.5 spatial fields and coordinates | Included |
  | 3.3 | `figure5_kde_curves.csv`; `figure5_wasserstein.csv` | PM2.5 distribution curves and Wasserstein distances | Included |

- `figures/O3_fine_tuned`

  | Num | Folder | Comments | How to get it? |
  | --- | ------ | -------- | -------------- |
  | 4.1 | `canonical_metrics.csv` | Aggregate O3 evaluation metrics | Included |
  | 4.2 | `data_efficiency.csv` | Transfer-learning performance across O3 training-data amounts | Included |
  | 4.3 | `figure10_monthly_metrics.csv` | Monthly O3 RMSE and R2 | Included |

## Acknowledgments

- We thank the developers of **CorrDiff, NVIDIA PhysicsNeMo, CMAQ, and WRF** for the modeling tools used in this work.
- We acknowledge the providers of **ERA5, CORINE Land Cover 2018, and SSP population data**.
