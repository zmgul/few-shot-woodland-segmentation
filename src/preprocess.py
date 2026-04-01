"""
LandCover.ai v1 Data Preprocessing
=================================================
Outputs:
  image_level_split.json  → image metadata + split + tile lists
  tile_registry.json      → per-class pixel statistics for each tile

Usage:
  python -m src.preprocess
  python -m src.preprocess --kfold
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from tqdm import tqdm

from src.config import cfg


def get_image_list():
    images = sorted(cfg.images_dir.glob("*.tif"))
    if not images:
        sys.exit(f"[ERROR] No .tif files found under {cfg.images_dir}.")
    print(f"{len(images)} images found.")
    return images


def compute_image_metadata(img_path, mask_path):
    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))
    h, w = img.shape[:2]
    total = h * w

    class_ratios = {}
    for cls_id in cfg.CLASS_NAMES:
        class_ratios[cls_id] = round(int(np.sum(mask == cls_id)) / total, 6)

    return {
        "name": img_path.stem,
        "height": h,
        "width": w,
        "res_group": "50cm" if max(h, w) < cfg.RESOLUTION_THRESHOLD else "25cm",
        "class_ratios": class_ratios,
    }


def create_primary_split(metadata_list):
    names = np.array([m["name"] for m in metadata_list])
    res_keys = np.array([m["res_group"] for m in metadata_list])
    train_r, val_r, test_r = cfg.SPLIT_RATIO

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_r + test_r, random_state=cfg.RANDOM_SEED,
    )
    train_idx, valtest_idx = next(sss.split(names, res_keys))

    rng = np.random.RandomState(cfg.RANDOM_SEED)
    valtest_shuffled = rng.permutation(valtest_idx)
    n_test = max(1, round(len(valtest_shuffled) * test_r / (val_r + test_r)))
    test_idx = valtest_shuffled[:n_test]
    val_idx = valtest_shuffled[n_test:]

    split = {
        "train": sorted(names[train_idx].tolist()),
        "val":   sorted(names[val_idx].tolist()),
        "test":  sorted(names[test_idx].tolist()),
    }

    print("\n[PRIMARY SPLIT]")
    for k, v in split.items():
        res = [next(m for m in metadata_list if m["name"] == n)["res_group"] for n in v]
        n25 = sum(1 for r in res if r == "25cm")
        print(f"  {k:5s}: {len(v)} images (25cm:{n25}, 50cm:{len(v)-n25})")
    return split


def create_kfold_splits(metadata_list):
    names = np.array([m["name"] for m in metadata_list])
    res_keys = np.array([m["res_group"] for m in metadata_list])

    skf = StratifiedKFold(
        n_splits=cfg.KFOLD_N_SPLITS, shuffle=True, random_state=cfg.RANDOM_SEED,
    )
    folds = {}
    print(f"\n[K-FOLD] {cfg.KFOLD_N_SPLITS} folds")
    for i, (tr, te) in enumerate(skf.split(names, res_keys)):
        folds[f"fold_{i}"] = {
            "train": sorted(names[tr].tolist()),
            "test":  sorted(names[te].tolist()),
        }
        print(f"  Fold {i}: train={len(tr)}, test={len(te)}")
    return folds


def tile_single_image(img_path, mask_path, out_dir):
    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path))
    h, w = img.shape[:2]
    stem = img_path.stem
    ts = cfg.TILE_SIZE

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    tiles = []
    ri = 0
    for y in range(0, h, ts):
        ci = 0
        for x in range(0, w, ts):
            if cfg.DROP_INCOMPLETE_TILES and (y + ts > h or x + ts > w):
                ci += 1; continue
            name = f"{stem}_r{ri:03d}_c{ci:03d}"
            Image.fromarray(img[y:y+ts, x:x+ts]).save(out_dir / "images" / f"{name}.png")
            Image.fromarray(mask[y:y+ts, x:x+ts]).save(out_dir / "masks" / f"{name}.png")
            tiles.append(name)
            ci += 1
        ri += 1
    return tiles


def tile_all_splits(split_dict, metadata_dict):
    result = {}
    for sn, img_names in split_dict.items():
        out_dir = cfg.tiles_dir / sn
        all_tiles = []

        print(f"\n[TILING] {sn} ({len(img_names)} images)...")
        for name in tqdm(img_names, desc=f"  {sn}"):
            ip = cfg.images_dir / f"{name}.tif"
            mp = cfg.masks_dir / f"{name}.tif"
            if not ip.exists() or not mp.exists():
                print(f"  [WARNING] {name} skipped.")
                continue
            for t in tile_single_image(ip, mp, out_dir):
                all_tiles.append({"tile_name": t, "source_image": name})

        result[sn] = all_tiles
        print(f"  → {len(all_tiles)} tiles")
    return result


def build_tile_registry(tile_info_by_split):
    """
    Records per-class pixel count/ratio for each tile.
    Does NOT perform filtering.

    Format:
      { "splits": { "train": { "tile_name": { "source_image": "...",
          "class_stats": { "0": {"pixel_count": N, "ratio": R}, ... } } } } }
    """
    registry = {"splits": {}}

    for sn, tile_list in tile_info_by_split.items():
        mask_dir = cfg.tiles_dir / sn / "masks"
        if not mask_dir.exists():
            continue

        split_tiles = {}
        print(f"\n[REGISTRY] {sn} ({len(tile_list)} tile)...")
        for ti in tqdm(tile_list, desc=f"  {sn}"):
            tn = ti["tile_name"]
            mp = mask_dir / f"{tn}.png"
            if not mp.exists():
                continue

            mask = np.array(Image.open(mp))
            total = mask.size
            stats = {}
            for cls_id in cfg.CLASS_NAMES:
                count = int(np.sum(mask == cls_id))
                stats[str(cls_id)] = {
                    "pixel_count": count,
                    "ratio": round(count / total, 6),
                }

            split_tiles[tn] = {
                "source_image": ti["source_image"],
                "class_stats": stats,
            }

        registry["splits"][sn] = split_tiles
        print(f"  → {len(split_tiles)} tiles recorded.")

    return registry


def main():
    parser = argparse.ArgumentParser(description="LandCover.ai v1 — Preprocessing")
    parser.add_argument("--kfold", action="store_true")
    args = parser.parse_args()

    Path(cfg.DATA_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Metadata
    paths = get_image_list()
    print("\n[1/4] Metadata...")
    meta = [compute_image_metadata(p, cfg.masks_dir / p.name)
            for p in tqdm(paths, desc="  Metadata")]
    meta_dict = {m["name"]: m for m in meta}

    wd = [m["class_ratios"][cfg.NOVEL_CLASS] for m in meta]
    print(f"  Woodland: min={min(wd):.3f}, max={max(wd):.3f}, mean={np.mean(wd):.3f}")

    # 2. Split
    print("\n[2/4] Image-level split...")
    primary = create_primary_split(meta)
    kfold = create_kfold_splits(meta) if args.kfold else None

    # 3. Tiling
    print("\n[3/4] Tiling...")
    tiles = tile_all_splits(primary, meta_dict)

    # 4. Registry
    print("\n[4/4] Tile registry...")
    registry = build_tile_registry(tiles)

    # Save
    split_out = {
        "metadata": {m["name"]: m for m in meta},
        "primary_split": {"image_names": primary, "tiles": tiles},
    }
    if kfold:
        split_out["kfold_splits"] = kfold

    with open(cfg.split_file, "w", encoding="utf-8") as f:
        json.dump(split_out, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {cfg.split_file}")

    with open(cfg.tile_registry, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {cfg.tile_registry}")

    # Summary
    print("\n" + "=" * 50)
    print(f"  {'Set':<8} {'Images':>10} {'Tiles':>8}")
    print(f"  {'-' * 30}")
    total = 0
    for s in ["train", "val", "test"]:
        n = len(tiles[s])
        total += n
        print(f"  {s:<8} {len(primary[s]):>10} {n:>8}")
    print(f"  {'-' * 30}")
    print(f"  {'TOTAL':<8} {len(meta):>10} {total:>8}")
    print("=" * 50)


if __name__ == "__main__":
    main()
