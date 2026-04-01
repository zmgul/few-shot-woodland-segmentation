from pathlib import Path
from pydantic import BaseModel, Field, PositiveInt, field_validator
from typing import Dict, Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WoodlandConfig(BaseModel):

    # ══════════════════════════════════════════════════════════
    #  PATHS
    # ══════════════════════════════════════════════════════════

    # RAW_DIR: preprocess.py reads raw images from here (images/, masks/).
    # Not used during training after preprocessing, but required if
    # preprocessing needs to be re-run.
    RAW_DIR: str = str(PROJECT_ROOT / "data" / "raw" / "landcover.ai.v1")

    DATA_DIR: str = str(PROJECT_ROOT / "data" / "processed")
    FIGURES_DIR: str = str(PROJECT_ROOT / "figures")
    EXPERIMENTS_DIR: str = str(PROJECT_ROOT / "experiments")

    # ══════════════════════════════════════════════════════════
    #  TILING (Boguszewski 2021 protocol)
    # ══════════════════════════════════════════════════════════

    TILE_SIZE: PositiveInt = 512        # Matches the original protocol exactly
    DROP_INCOMPLETE_TILES: bool = True  # Drop edge tiles

    # ══════════════════════════════════════════════════════════
    #  SPLIT (runs once, then read from JSON)
    # ══════════════════════════════════════════════════════════

    SPLIT_RATIO: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    RANDOM_SEED: int = 42
    RESOLUTION_THRESHOLD: int = 5000  # long edge < 5000 → 50cm

    # ══════════════════════════════════════════════════════════
    #  TILE FILTERING (threshold analysis)
    # ══════════════════════════════════════════════════════════

    # Theoretical lower bound: 36 FG locations on a 60×60 feature map → 1%
    # Per-class thresholds:
    #   Building 0.5%: rare class (0.9%), higher threshold collapses image diversity
    #   Woodland 5.0%: abundant class (33%), high threshold for quality prototypes
    #   Water 1.0%: moderate prevalence, matches theoretical lower bound
    #   Road 1.0%: narrow structure, naturally low ratio
    MIN_FG: Dict[int, float] = {1: 0.005, 2: 0.05, 3: 0.01, 4: 0.01}

    # ══════════════════════════════════════════════════════════
    #  CLASSES (dataset-dependent)
    # ══════════════════════════════════════════════════════════

    BASE_CLASSES: List[int] = [1, 3, 4]  # Training: building, water, road
    NOVEL_CLASS: int = 2                  # Test: woodland
    CLASS_NAMES: Dict[int, str] = {
        0: "Background", 1: "Building", 2: "Woodland", 3: "Water", 4: "Road"
    }

    # ══════════════════════════════════════════════════════════
    #  NORMALIZATION (tied to frozen backbone)
    # ══════════════════════════════════════════════════════════

    # ImageNet statistics. Backbone is frozen and trained on ImageNet —
    # it must receive data in its expected distribution. Cannot be changed.
    IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

    # PFENet/HSNet standard. ResNet-50 with stride=8 produces 60×60 feature maps.
    # 512 could also be used (64×64 feature map) but 473 for literature consistency.
    INPUT_SIZE: PositiveInt = 473

    # ══════════════════════════════════════════════════════════
    #  AUGMENTATION (domain-specific choices)
    # ══════════════════════════════════════════════════════════

    # ±10%: preserves spectral information in aerial imagery (green tone = vegetation type).
    # >30% destroys spectral information. Frozen backbone already has
    # basic robustness from ImageNet diversity.
    COLOR_JITTER: Dict[str, float] = {
        "brightness": 0.1, "contrast": 0.1, "saturation": 0.1
    }

    # ══════════════════════════════════════════════════════════
    #  K-FOLD 
    # ══════════════════════════════════════════════════════════

    KFOLD_N_SPLITS: PositiveInt = 5

    # ══════════════════════════════════════════════════════════
    #  EPISODIC TRAINING
    # ══════════════════════════════════════════════════════════

    # K_SUPPORT (FIXED): 5-shot. Project definition.
    K_SUPPORT: PositiveInt = 5
    N_QUERY: PositiveInt = 1

    # EPISODES_PER_EPOCH (FIXED):
    # Train tile pool: building(1040) + water(1109) + road(2457) = 4606 tiles
    # 5-shot episode consumes 6 tiles → 1 full pass = 4606/6 ≈ 768 episodes
    # 2000 episodes → pool scanned ~2.6 times → sufficient diversity ensured
    EPISODES_PER_EPOCH: PositiveInt = 2000

    # TOTAL_EPISODES (VARIABLE — stops automatically with early stopping):
    # FSS literature: PANet 30k, PFENet 50k, HSNet 50k
    # 50,000 chosen (PFENet/HSNet standard). Early stopping
    # typically triggers at 20-30k, 50k is the upper bound.
    TOTAL_EPISODES: PositiveInt = 50000

    # VAL_INTERVAL (VARIABLE):
    # Validation after 1000 episodes = 250 batches (bs=4) of training
    # 50,000 / 1,000 = 50 validation points → sufficient granularity
    # More frequent (500) → higher validation cost
    # Less frequent (2000) → late detection of improvement
    VAL_INTERVAL: PositiveInt = 1000

    # TEST_EPISODES (VARIABLE):
    # Evaluation over 2000 episodes.
    # With estimated IoU std ~0.15, standard error = 0.15/sqrt(2000) = 0.0034
    # 95% confidence interval: ±0.66% → sufficient statistical reliability
    # iSAID-5i: 5000, PASCAL-5i: used in the 1000-5000 range
    TEST_EPISODES: PositiveInt = 2000

    # N_SEEDS (FIXED):
    # Results reported as mean ± std. FSS standard:
    # PANet 5, PFENet 3, HSNet 3. 3 chosen.
    N_SEEDS: PositiveInt = 3

    # ══════════════════════════════════════════════════════════
    #  MODEL (VARIABLE — experiment parameters)
    # ══════════════════════════════════════════════════════════

    # BACKBONE (VARIABLE — Phase 1 experiment parameter):
    # resnet18, resnet34, resnet50, resnet101, resnet152 → torchvision
    # xception → timm (entry/middle/exit, 2048 channels)
    # swin_t   → torchvision (Shifted Window Transformer, 768 channels)
    # vit_b16  → timm (Vision Transformer B/16, 768 channels)
    BACKBONE: str = "resnet50"

    # PRETRAINED (VARIABLE — Phase 2 experiment parameter):
    # imagenet_v1:  torchvision IMAGENET1K_V1 (same as Phase 1)
    # million_aid:  ResNet-50 trained on Million-AID
    # bigearthnet:  ResNet-50 trained on BigEarthNet (13-band → RGB)
    # seco:         ResNet-50 trained on SeCo (13-band → RGB)
    # cityscapes:   ResNet-50 trained on Cityscapes
    PRETRAINED: str = "imagenet_v1"

    # External pretrained checkpoint directory (except imagenet_v1)
    PRETRAINED_CKPT_DIR: str = str(PROJECT_ROOT / "experiments" / "pretrained_weights")

    # FEATURE_DIM: Backbone 2048 → reduced to this dimension.
    # 256: dimension used in HSNet and PFENet.
    # Lower (128) → information loss. Higher (512) → curse of dimensionality.
    FEATURE_DIM: PositiveInt = 256

    # INITIAL_TEMPERATURE: Cosine similarity in [-1,1] range.
    # Too narrow for sigmoid/cross-entropy → scaled with temperature.
    # 10.0 initial value, learned during training (log-space).
    INITIAL_TEMPERATURE: float = 10.0

    # UNFREEZE_FROM (VARIABLE — main experiment parameter):
    # "none"   → Experiment A: entire backbone frozen, ~525K trainable params
    # "layer4" → Experiment B: layer4 unfrozen, ~7.6M trainable params
    # "layer3" → Experiment C: layer3+4 unfrozen, ~14.5M trainable params
    UNFREEZE_FROM: str = "layer3"

    # ══════════════════════════════════════════════════════════
    #  TRAINING HYPERPARAMETERS
    # ══════════════════════════════════════════════════════════

    # BATCH_SIZE (VARIABLE — depends on GPU memory):
    # 4 episodes/batch → 24 images (6 tiles × 4 episodes)
    # Fits comfortably on A100 80GB. 8 can also be tried.
    # Small batch → noisy gradients → better generalization (regularization effect)
    BATCH_SIZE: PositiveInt = 4

    # LR (VARIABLE):
    # 1e-4 is the standard starting point with AdamW.
    # FSS works using SGD use higher values (PANet 1e-3, PFENet 5e-3)
    # because Adam applies adaptive lr, a lower starting point suffices.
    # Decays to 1e-6 with cosine annealing.
    LR: float = Field(default=1e-4, gt=0, lt=1)

    # BACKBONE_LR_FACTOR (VARIABLE — for unfreeze experiments):
    # LR multiplier for unfrozen backbone layers.
    # 0.1 → backbone lr = 1e-5. Low to preserve pretrained weights.
    # Has no effect when UNFREEZE_FROM="none" (backbone frozen).
    BACKBONE_LR_FACTOR: float = Field(default=0.1, gt=0, le=1)

    # WEIGHT_DECAY (FIXED):
    # AdamW L2 regularization. 1e-4 standard.
    WEIGHT_DECAY: float = Field(default=1e-4, ge=0)

    # GRAD_CLIP (FIXED):
    # Gradient norm clipping. 5.0 safety net.
    # Cosine similarity + temperature → low risk of gradient explosion.
    # Not triggered during normal training, protects against anomalies.
    GRAD_CLIP: float = 5.0

    # NUM_WORKERS (VARIABLE — hardware-dependent):
    # 20+ CPU cores on HPC node → 8 workers sufficient.
    # Each worker loads PNG + applies augmentation.
    # Too high (>16) → filesystem bottleneck.
    # 4 on local machine, 8 on server.
    NUM_WORKERS: int = Field(default=8, ge=0)

    # ══════════════════════════════════════════════════════════
    #  LOSS (VARIABLE)
    # ══════════════════════════════════════════════════════════

    # Focal Loss: focuses on hard pixels.
    # alpha=0.25: FG/BG imbalance weight (Lin et al., 2017)
    # gamma=2.0: reduces contribution of easy examples (original paper value)
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0

    # DICE_WEIGHT: loss = focal + dice_weight × dice
    # 0.5: slightly less weight to Dice without equal weighting.
    # Focal is pixel-based, Dice is region-based → more stable together.
    DICE_WEIGHT: float = 0.5

    # ══════════════════════════════════════════════════════════
    #  MLFLOW
    # ══════════════════════════════════════════════════════════

    MLFLOW_TRACKING_URI: str = str(PROJECT_ROOT / "mlruns")
    MLFLOW_EXPERIMENT_NAME: str = "woodland-fss"

    # ══════════════════════════════════════════════════════════
    #  DERIVED PATHS
    # ══════════════════════════════════════════════════════════

    @property
    def images_dir(self) -> Path:
        return Path(self.RAW_DIR) / "images"

    @property
    def masks_dir(self) -> Path:
        return Path(self.RAW_DIR) / "masks"

    @property
    def tiles_dir(self) -> Path:
        return Path(self.DATA_DIR) / "tiles"

    @property
    def split_file(self) -> Path:
        return Path(self.DATA_DIR) / "image_level_split.json"

    @property
    def tile_registry(self) -> Path:
        return Path(self.DATA_DIR) / "tile_registry.json"

    @field_validator("SPLIT_RATIO")
    @classmethod
    def validate_split(cls, v):
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError("SPLIT_RATIO must sum to 1.0!")
        return v

    @field_validator("BACKBONE")
    @classmethod
    def validate_backbone(cls, v):
        valid = {
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            "xception", "swin_t", "vit_b16",
        }
        if v not in valid:
            raise ValueError(f"Invalid backbone: '{v}'. Options: {sorted(valid)}")
        return v

    @field_validator("PRETRAINED")
    @classmethod
    def validate_pretrained(cls, v):
        valid = {
            "imagenet_v1", "million_aid", "bigearthnet", "seco", "cityscapes",
        }
        if v not in valid:
            raise ValueError(f"Invalid pretrained: '{v}'. Options: {sorted(valid)}")
        return v

    @field_validator("UNFREEZE_FROM")
    @classmethod
    def validate_unfreeze(cls, v):
        if v not in ("none", "layer4", "layer3"):
            raise ValueError(f"Invalid: {v}. Options: none, layer4, layer3")
        return v


cfg = WoodlandConfig()
