"""
dataset.py — Few-Shot Semantic Segmentation Dataset & Transforms
=================================================================
FSS Protocol:
  TRAIN → base classes (building, water, road). Woodland = background.
  VAL/TEST → novel class (woodland). Model encounters it for the first time.

5-Fold Cross-Validation:
  When fold_i is specified, kfold_splits from image_level_split.json is used.
  Tiles are read cross-directory from physical directories (not moved).
  20% val is split from fold train (deterministic: seed + fold_i).

Usage:
  from src.dataset import WoodlandFewShotDataset
  from src.dataset import get_train_transform, get_val_transform
"""

import json
import random
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

from src.config import cfg


# ================================================================
#  TRANSFORMS
# ================================================================

class PairCompose:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, image, mask):
        for transform in self.transform_list:
            image, mask = transform(image, mask)
        return image, mask


class PairResize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class PairRandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, mask):
        if random.random() < self.probability:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class PairRandomVerticalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, mask):
        if random.random() < self.probability:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class PairRandomRotation90:
    def __call__(self, image, mask):
        rotation_count = random.choice([0, 1, 2, 3])
        if rotation_count > 0:
            angle = rotation_count * 90
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class PairColorJitter:
    def __init__(self, **kwargs):
        self.jitter = transforms.ColorJitter(**kwargs)

    def __call__(self, image, mask):
        return self.jitter(image), mask


class PairToTensor:
    def __call__(self, image, mask):
        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        return image_tensor, mask_tensor


class PairNormalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean or cfg.IMAGENET_MEAN
        self.std = std or cfg.IMAGENET_STD

    def __call__(self, image, mask):
        return TF.normalize(image, self.mean, self.std), mask


def get_train_transform(input_size=None):
    size = input_size or cfg.INPUT_SIZE
    return PairCompose([
        PairResize(size),
        PairRandomHorizontalFlip(),
        PairRandomVerticalFlip(),
        PairRandomRotation90(),
        PairColorJitter(**cfg.COLOR_JITTER),
        PairToTensor(),
        PairNormalize(),
    ])


def get_val_transform(input_size=None):
    size = input_size or cfg.INPUT_SIZE
    return PairCompose([
        PairResize(size),
        PairToTensor(),
        PairNormalize(),
    ])


# ================================================================
#  DATASET
# ================================================================

class WoodlandFewShotDataset(Dataset):
    """
    TRAIN: a random base class (building, water, road) is selected per episode.
           Woodland pixels become 0 in binary masks → model never sees them.
    VAL/TEST: episodes are created on the novel class (woodland).

    fold_i=0..4: image list is retrieved from image_level_split.json.
    Tiles are read cross-directory from physical directories (not moved).
    """

    def __init__(self, split="train", fold_i=0, k_shot=None,
                 episodes_per_epoch=None, transform=None, seed=None):
        super().__init__()
        self.split = split
        self.fold_i = fold_i
        self.k_shot = k_shot or cfg.K_SUPPORT
        self.episodes_per_epoch = episodes_per_epoch or cfg.EPISODES_PER_EPOCH
        self.transform = transform
        self.is_train = (split == "train")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        with open(cfg.tile_registry, "r", encoding="utf-8") as f:
            registry = json.load(f)

        target_images = self._get_kfold_images(split, fold_i)

        # Merge tiles from all physical splits (cross-directory)
        all_tile_stats = {}
        self._tile_to_dir = {}
        for phys_split in ("train", "val", "test"):
            for tile_name, tile_data in registry["splits"].get(phys_split, {}).items():
                all_tile_stats[tile_name] = tile_data
                self._tile_to_dir[tile_name] = phys_split

        # Filter tiles belonging to target images
        filtered_stats = {
            name: data for name, data in all_tile_stats.items()
            if data["source_image"] in target_images
        }

        # ── Build per-class tile pool ──────────────────────────────────
        if self.is_train:
            active_classes = cfg.BASE_CLASSES
        else:
            active_classes = [cfg.NOVEL_CLASS]

        self.class_pool = {}

        for class_id in active_classes:
            minimum_fg_ratio = cfg.MIN_FG.get(class_id, 0.01)

            valid_tiles = []
            source_to_indices = defaultdict(list)

            for tile_name, tile_data in filtered_stats.items():
                fg_ratio = tile_data["class_stats"][str(class_id)]["ratio"]
                if fg_ratio >= minimum_fg_ratio:
                    tile_index = len(valid_tiles)
                    valid_tiles.append({
                        "tile_name": tile_name,
                        "source_image": tile_data["source_image"],
                    })
                    source_to_indices[tile_data["source_image"]].append(tile_index)

            source_list = list(source_to_indices.keys())

            if len(valid_tiles) < self.k_shot + 1:
                print(f"  [WARNING] {cfg.CLASS_NAMES[class_id]}: "
                      f"{len(valid_tiles)} tiles — insufficient")
                continue

            if len(source_list) < 2:
                print(f"  [WARNING] {cfg.CLASS_NAMES[class_id]}: single source image")
                continue

            self.class_pool[class_id] = {
                "valid_tiles": valid_tiles,
                "source_to_indices": dict(source_to_indices),
                "source_list": source_list,
            }

        if not self.class_pool:
            raise ValueError(f"{split} (fold={fold_i}): not enough tiles for any class!")

        mode_label = "BASE (woodland hidden)" if self.is_train else "NOVEL (woodland)"
        print(f"\n[DATASET] {split}/fold{fold_i} — {mode_label} — k={self.k_shot}")
        print(f"  Images ({len(target_images)}): {sorted(target_images)}")
        for class_id, pool in self.class_pool.items():
            print(f"  {cfg.CLASS_NAMES[class_id]:<14}: "
                  f"{len(pool['valid_tiles'])} tile, "
                  f"{len(pool['source_list'])} images")

    def _get_kfold_images(self, split, fold_i):
        """
        Returns the target image set for fold_i from image_level_split.json.
        kfold_splits["fold_i"]["train"] → 80% train, 20% val (deterministic)
        kfold_splits["fold_i"]["test"]  → test
        """
        with open(cfg.split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        fold_data = split_data["kfold_splits"][f"fold_{fold_i}"]
        all_train = fold_data["train"]   # list of image names
        test_set = set(fold_data["test"])

        if split == "test":
            return test_set

        # Val: round(20%), deterministic seed + fold_i
        rng = np.random.RandomState(cfg.RANDOM_SEED + fold_i)
        shuffled = list(all_train)
        rng.shuffle(shuffled)
        n_val = max(1, round(0.2 * len(shuffled)))

        if split == "val":
            return set(shuffled[:n_val])
        else:  # train
            return set(shuffled[n_val:])

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, index):
        target_class = random.choice(list(self.class_pool.keys()))
        pool = self.class_pool[target_class]
        valid_tiles = pool["valid_tiles"]
        source_to_indices = pool["source_to_indices"]
        source_list = pool["source_list"]

        query_source = random.choice(source_list)
        query_index = random.choice(source_to_indices[query_source])
        query_info = valid_tiles[query_index]

        other_sources = [s for s in source_list if s != query_source]
        support_infos = []
        if len(other_sources) >= self.k_shot:
            for source in random.sample(other_sources, self.k_shot):
                support_infos.append(valid_tiles[random.choice(source_to_indices[source])])
        else:
            for _ in range(self.k_shot):
                source = random.choice(other_sources) if other_sources else query_source
                support_infos.append(valid_tiles[random.choice(source_to_indices[source])])

        query_image, query_mask = self._load_tile(query_info["tile_name"], target_class)
        support_pairs = [
            self._load_tile(info["tile_name"], target_class) for info in support_infos
        ]

        if self.transform:
            query_image, query_mask = self.transform(query_image, query_mask)
            support_pairs = [self.transform(img, msk) for img, msk in support_pairs]

        return {
            "support_images": torch.stack([pair[0] for pair in support_pairs]),
            "support_masks":  torch.stack([pair[1] for pair in support_pairs]),
            "query_image":    query_image,
            "query_mask":     query_mask,
            "target_class":   target_class,
            "query_source":   query_info["source_image"],
            "support_sources": [info["source_image"] for info in support_infos],
        }

    def _load_tile(self, tile_name, target_class):
        """Loads the tile cross-directory from its physical directory."""
        split_directory = cfg.tiles_dir / self._tile_to_dir[tile_name]
        image = Image.open(split_directory / "images" / f"{tile_name}.png").convert("RGB")
        mask_array = np.array(Image.open(split_directory / "masks" / f"{tile_name}.png"))
        binary_mask = Image.fromarray((mask_array == target_class).astype(np.uint8))
        return image, binary_mask


# ================================================================
#  PROTOTYPE EXTRACTION
# ================================================================

def compute_fg_prototype(features, mask, epsilon=1e-5):
    """Foreground prototype: mean of mask=1 regions."""
    batch_size, channels, fh, fw = features.shape
    resized_mask = torch.nn.functional.interpolate(
        mask.unsqueeze(1).float(), size=(fh, fw),
        mode="bilinear", align_corners=False)
    fg_mask = (resized_mask > 0.5).float()
    return (features * fg_mask).sum(dim=(2, 3)) / (fg_mask.sum(dim=(2, 3)) + epsilon)


def compute_bg_prototype(features, mask, epsilon=1e-5):
    """Background prototype: mean of mask=0 regions."""
    batch_size, channels, fh, fw = features.shape
    resized_mask = torch.nn.functional.interpolate(
        mask.unsqueeze(1).float(), size=(fh, fw),
        mode="bilinear", align_corners=False)
    bg_mask = 1.0 - (resized_mask > 0.5).float()
    return (features * bg_mask).sum(dim=(2, 3)) / (bg_mask.sum(dim=(2, 3)) + epsilon)
