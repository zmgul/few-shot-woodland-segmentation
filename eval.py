"""
eval.py — Best Model ile Final Test Değerlendirmesi (5-Fold CV)
===============================================================
Checkpoint yolunu config'deki BACKBONE, UNFREEZE_FROM, fold_i ve
RANDOM_SEED'den otomatik türetir. train.py ile aynı config'i kullanır.

Kullanım:
  python eval.py

Akış: config değiştir → python train.py → python eval.py
"""

from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from src.config import cfg
from src.model import ProtoNet, BACKBONE_LABELS, PRETRAINED_LABELS
from src.dataset import WoodlandFewShotDataset, get_val_transform
from src.utils import compute_metrics


def get_checkpoint_path(backbone_label, pretrained_label, unfreeze, fold_label, seed):
    run_name  = f"ProtoNet_{backbone_label}_{pretrained_label}_{unfreeze}_{fold_label}_seed-{seed}"
    ckpt_path = Path(cfg.EXPERIMENTS_DIR) / "checkpoints" / f"best_{run_name}.pt"
    return ckpt_path


def eval_fold(fold_i, seed, device):
    """fold_i=0..4 → kfold CV."""

    fold_label      = f"fold{fold_i}"
    backbone_label  = BACKBONE_LABELS[cfg.BACKBONE]
    pretrained_label = PRETRAINED_LABELS[cfg.PRETRAINED]
    checkpoint_path = get_checkpoint_path(backbone_label, pretrained_label, cfg.UNFREEZE_FROM, fold_label, seed)

    if not checkpoint_path.exists():
        print(f"\n[UYARI] Checkpoint bulunamadı: {checkpoint_path}")
        print("Mevcut checkpoint'lar:")
        ckpt_dir = Path(cfg.EXPERIMENTS_DIR) / "checkpoints"
        if ckpt_dir.exists():
            for f in sorted(ckpt_dir.glob("best_*.pt")):
                print(f"  {f.name}")
        else:
            print("  (checkpoints klasörü yok — önce train.py çalıştır)")
        return None

    print(f"\n── {fold_label} ──")
    checkpoint   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", {})

    unfreeze      = saved_config.get("UNFREEZE_FROM", cfg.UNFREEZE_FROM)
    feature_dim   = saved_config.get("FEATURE_DIM",   cfg.FEATURE_DIM)
    backbone_name = saved_config.get("BACKBONE",      cfg.BACKBONE)
    input_size    = saved_config.get("INPUT_SIZE",     cfg.INPUT_SIZE)
    pretrained    = saved_config.get("PRETRAINED",     cfg.PRETRAINED)

    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Episode:    {checkpoint.get('episode', '?')}")
    print(f"  Val mIoU:   {checkpoint.get('val_mIoU', 0):.4f}")

    # ── Model ──
    model = ProtoNet(
        backbone_name=backbone_name,
        feature_dim=feature_dim,
        unfreeze_from=unfreeze,
        input_size=input_size,
        pretrained=pretrained,
        pretrained_ckpt_dir=cfg.PRETRAINED_CKPT_DIR,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Test veri seti ──
    test_dataset = WoodlandFewShotDataset(
        split="test",
        fold_i=fold_i,
        k_shot=cfg.K_SUPPORT,
        episodes_per_epoch=cfg.TEST_EPISODES,
        transform=get_val_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    # ── Değerlendirme ──
    print(f"\n── Test ({cfg.TEST_EPISODES} episode) ──")
    total_metrics = {"fgIoU": 0, "mIoU": 0, "precision": 0, "recall": 0, "f1": 0}
    num_batches   = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Test/{fold_label}"):
            support_images = batch["support_images"].to(device)
            support_masks  = batch["support_masks"].to(device)
            query_image    = batch["query_image"].to(device)
            targets        = batch["query_mask"].unsqueeze(1).float().to(device)

            use_amp = cfg.BACKBONE.startswith("resnet")
            with autocast(enabled=use_amp):
                logits = model(support_images, support_masks, query_image)

            metrics = compute_metrics(logits.float(), targets)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1

    avg = {k: v / num_batches for k, v in total_metrics.items()}

    # ── MLflow ──
    run_name = f"TEST_ProtoNet_{backbone_label}_{pretrained_label}_{unfreeze}_{fold_label}_seed-{seed}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("phase",           "test")
        mlflow.log_param("model",           "ProtoNet")
        mlflow.log_param("backbone",        backbone_label)
        mlflow.log_param("pretrained",      pretrained_label)
        mlflow.log_param("UNFREEZE_FROM",   unfreeze)
        mlflow.log_param("fold",            fold_label)
        mlflow.log_param("RANDOM_SEED",     seed)
        mlflow.log_param("checkpoint",      checkpoint_path.name)
        mlflow.log_param("test_episodes",   cfg.TEST_EPISODES)
        mlflow.log_param("train_episode",   checkpoint.get("episode", 0))
        mlflow.log_param("val_mIoU_at_save", f"{checkpoint.get('val_mIoU', 0):.4f}")
        for key, value in avg.items():
            mlflow.log_metric(f"test_{key}", value)

    # ── Rapor ──
    print(f"  fgIoU:     {avg['fgIoU']:.4f}")
    print(f"  mIoU:      {avg['mIoU']:.4f}")
    print(f"  Precision: {avg['precision']:.4f}")
    print(f"  Recall:    {avg['recall']:.4f}")
    print(f"  F1:        {avg['f1']:.4f}")

    return avg


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    folds_to_run = list(range(cfg.KFOLD_N_SPLITS))

    seed   = cfg.RANDOM_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)

    backbone_label = BACKBONE_LABELS[cfg.BACKBONE]
    results = []
    for fold_i in folds_to_run:
        avg = eval_fold(fold_i, seed, device)
        if avg is not None:
            results.append(avg)

    if not results:
        print("\n[HATA] Hiçbir fold değerlendirilemedi.")
        return

    # ── Mean±std: MLflow + konsol ──
    metrics_keys = list(results[0].keys())
    pretrained_label = PRETRAINED_LABELS[cfg.PRETRAINED]
    summary_run_name = (
        f"SUMMARY_ProtoNet_{backbone_label}_{pretrained_label}_{cfg.UNFREEZE_FROM}_seed-{seed}"
    )
    with mlflow.start_run(run_name=summary_run_name):
        mlflow.log_param("phase",         "test_summary")
        mlflow.log_param("model",         "ProtoNet")
        mlflow.log_param("backbone",      backbone_label)
        mlflow.log_param("pretrained",    pretrained_label)
        mlflow.log_param("UNFREEZE_FROM", cfg.UNFREEZE_FROM)
        mlflow.log_param("RANDOM_SEED",   seed)
        mlflow.log_param("n_folds",       len(results))
        for key in metrics_keys:
            vals = np.array([r[key] for r in results])
            mlflow.log_metric(f"test_{key}_mean", float(vals.mean()))
            mlflow.log_metric(f"test_{key}_std",  float(vals.std()))

    print(f"\n{'='*60}")
    print(f"  TEST SONUÇLARI — {backbone_label} / {pretrained_label} / {cfg.UNFREEZE_FROM}")
    print(f"  (mean ± std, {len(results)} fold)")
    print(f"{'='*60}")
    for key in metrics_keys:
        vals = np.array([r[key] for r in results])
        print(f"  {key:<12}: {vals.mean():.4f} ± {vals.std():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
