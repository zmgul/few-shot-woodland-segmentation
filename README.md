# Few-Shot Woodland Segmentation

**Master's thesis:** Few-shot semantic segmentation of woodlands in high-resolution aerial imagery.

Prototypical network (ProtoNet) based **1-way 5-shot binary segmentation** on [LandCover.ai v1](https://landcover.ai/), with woodland as the novel class.


## Qualitative Results

### Backbone Comparison
<p align="center">
  <img src="figures/qualitative_results/qualitative_backbone.png" width="900">
</p>

### Layer Unfreezing Effect
<p align="center">
  <img src="figures/qualitative_results/qualitative_unfreeze.png" width="900">
</p>

> Color coding: 🟢 True Positive | 🔴 False Positive | 🔵 False Negative

## Project Structure

```
woodland/
├── src/
│   ├── __init__.py
│   ├── config.py          # Pydantic config, all parameters
│   ├── dataset.py         # 5-fold CV, cross-directory tile loading
│   ├── model.py           # ProtoNet, dilated backbones (WIP)
│   ├── utils.py           # FocalLoss, DiceLoss, metrics
│   └── preprocess.py      # One-time: tiling + registry + kfold splits
├── train.py               # Episode-based training, MLflow logging (WIP)
├── eval.py                # 5-fold test evaluation (WIP)
├── data/                  # Not tracked (see Data section)
│   └── processed/
│       ├── image_level_split.json
│       ├── tile_registry.json
│       └── tiles/{train,val,test}/{images,masks}/
├── notebooks/
│   ├── EDA.ipynb          # Exploratory data analysis
│   ├── visualize.ipynb    # Qualitative result generation
│   └── training_curves.ipynb  # Training curve analysis
├── figures/
│   ├── qualitative_results/
│   ├── training_curves/
│   └── EDA/
├── Dockerfile             # PyTorch 2.1.0 + CUDA 11.8
├── run_altay.slurm        # HPC SLURM job script
└── pyproject.toml         # Dependencies (Poetry)
```

> **Note:** `data/` is not tracked in this repository. Dataset management via [DVC](https://dvc.org/) is planned.

## Dataset

[LandCover.ai v1](https://landcover.ai/) — 41 high-resolution aerial images (25–50 cm/pixel) over Poland.

| Class | Role | Pixel ratio |
|-------|------|-------------|
| Background | — | 58.1% |
| Building | Base (train) | 0.9% |
| **Woodland** | **Novel (test)** | **33.3%** |
| Water | Base (train) | 6.1% |
| Road | Base (train) | 1.6% |

## Tech Stack

| Component | Tool |
|-----------|------|
| Framework | PyTorch 2.1.0 |
| Pretrained weights | torchvision, timm |
| Experiment tracking | MLflow |
| Config management | Pydantic |
| Containerization | Docker → Apptainer (HPC) |
| HPC | NVIDIA A100 80GB, SLURM |

## Reproducibility

```bash
# Build container
docker build -t woodland .

# Run training (all 5 folds automatically)
python train.py

# Run evaluation
python eval.py
```

All experiments use `seed=42` for reproducibility. Results are reported as mean ± std over 5-fold cross-validation.
