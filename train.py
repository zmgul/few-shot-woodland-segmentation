"""
train.py — ProtoNet Episode-Bazlı Eğitim (MLflow + AMP)
==========================================================
Tüm parametreler config.py'den okunur.

Kullanım:
  python train.py

  MLflow UI:
  mlflow ui --backend-store-uri ./mlruns

Deney akışı (Aşama 1 örneği):
  config.py → BACKBONE = "resnet18", UNFREEZE_FROM = "none"
  python train.py   → 5 fold otomatik döner, 5 MLflow run oluşur
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow

from src.config import cfg
from src.model import ProtoNet, BACKBONE_LABELS, PRETRAINED_LABELS
from src.dataset import WoodlandFewShotDataset, get_train_transform, get_val_transform
from src.utils import FocalLoss, dice_loss, compute_metrics


# ══════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_infinite_loader(dataset, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    while True:
        yield from loader


def build_optimizer(model):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(layer in name for layer in ["layer0", "layer1", "layer2", "layer3", "layer4",
                                              "swin_stem", "swin_stage3", "swin_stage4"]):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [{"params": head_params, "lr": cfg.LR}]

    if backbone_params:
        backbone_lr = cfg.LR * cfg.BACKBONE_LR_FACTOR
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
        print(f"  Head LR: {cfg.LR}, Backbone LR: {backbone_lr}")
    else:
        print(f"  LR: {cfg.LR} (backbone frozen)")

    return torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)


# ══════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, val_loader, focal_criterion, device):
    model.eval()
    total_loss = 0
    total_metrics = {"fgIoU": 0, "mIoU": 0, "precision": 0, "recall": 0, "f1": 0}
    num_batches = 0

    for batch in val_loader:
        support_images = batch["support_images"].to(device)
        support_masks  = batch["support_masks"].to(device)
        query_image    = batch["query_image"].to(device)
        targets        = batch["query_mask"].unsqueeze(1).float().to(device)

        use_amp = cfg.BACKBONE.startswith("resnet")
        with autocast(enabled=use_amp):
            logits = model(support_images, support_masks, query_image)
            focal  = focal_criterion(logits, targets).mean()
            dice   = dice_loss(logits, targets).mean()
            loss   = focal + cfg.DICE_WEIGHT * dice

        metrics = compute_metrics(logits.float(), targets)

        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        num_batches += 1

    avg_loss    = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


# ══════════════════════════════════════════════════════════
#  TEK FOLD EĞİTİMİ
# ══════════════════════════════════════════════════════════

def run_fold(fold_i, seed, device):
    """fold_i=0..4 → kfold CV."""

    fold_label      = f"fold{fold_i}"
    backbone_label  = BACKBONE_LABELS[cfg.BACKBONE]
    pretrained_label = PRETRAINED_LABELS[cfg.PRETRAINED]
    run_name        = (
        f"ProtoNet_{backbone_label}_{pretrained_label}_{cfg.UNFREEZE_FROM}_{fold_label}_seed-{seed}"
    )

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    # ── Veri ──
    train_dataset = WoodlandFewShotDataset(
        split="train",
        fold_i=fold_i,
        k_shot=cfg.K_SUPPORT,
        episodes_per_epoch=cfg.EPISODES_PER_EPOCH,
        transform=get_train_transform(),
        seed=seed,
    )
    val_dataset = WoodlandFewShotDataset(
        split="val",
        fold_i=fold_i,
        k_shot=cfg.K_SUPPORT,
        episodes_per_epoch=cfg.TEST_EPISODES,
        transform=get_val_transform(),
        seed=seed,
    )

    train_iter = make_infinite_loader(train_dataset, cfg.BATCH_SIZE, cfg.NUM_WORKERS)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # ── Model ──
    model = ProtoNet(
        backbone_name=cfg.BACKBONE,
        feature_dim=cfg.FEATURE_DIM,
        initial_temperature=cfg.INITIAL_TEMPERATURE,
        unfreeze_from=cfg.UNFREEZE_FROM,
        input_size=cfg.INPUT_SIZE,
        pretrained=cfg.PRETRAINED,
        pretrained_ckpt_dir=cfg.PRETRAINED_CKPT_DIR,
    ).to(device)

    # ── Optimizer + Scheduler + AMP ──
    optimizer    = build_optimizer(model)
    total_steps  = cfg.TOTAL_EPISODES // cfg.BATCH_SIZE
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    scaler       = GradScaler()
    focal_criterion = FocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)

    # ── MLflow run ──
    with mlflow.start_run(run_name=run_name):

        # Config logla
        for key, value in cfg.model_dump().items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        mlflow.log_param("trainable_params", trainable)
        mlflow.log_param("frozen_params", frozen)
        mlflow.log_param("phase", "train")
        mlflow.log_param("model", "ProtoNet")
        mlflow.log_param("backbone", backbone_label)
        mlflow.log_param("pretrained", pretrained_label)
        mlflow.log_param("fold", fold_label)

        # ── Episode bazlı eğitim ──
        print(f"\n── Training ({cfg.TOTAL_EPISODES} episodes, "
              f"val every {cfg.VAL_INTERVAL}) ──\n")

        episode_count    = 0
        running_loss     = 0
        running_metrics  = {"fgIoU": 0, "mIoU": 0}
        running_count    = 0
        best_val_miou    = 0.0
        best_episode     = 0
        start_time       = time.time()
        last_val_episode = 0

        model.train()
        pbar = tqdm(total=cfg.TOTAL_EPISODES, desc=fold_label, unit="ep")

        while episode_count < cfg.TOTAL_EPISODES:
            batch = next(train_iter)

            support_images = batch["support_images"].to(device, non_blocking=True)
            support_masks  = batch["support_masks"].to(device, non_blocking=True)
            query_image    = batch["query_image"].to(device, non_blocking=True)
            targets        = batch["query_mask"].unsqueeze(1).float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            use_amp = cfg.BACKBONE.startswith("resnet")
            with autocast(enabled=use_amp):
                logits = model(support_images, support_masks, query_image)
                focal  = focal_criterion(logits, targets).mean()
                dice   = dice_loss(logits, targets).mean()
                loss   = focal + cfg.DICE_WEIGHT * dice

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
                optimizer.step()
            scheduler.step()

            batch_episodes  = support_images.shape[0]
            episode_count  += batch_episodes
            running_loss   += loss.item()
            running_count  += 1

            with torch.no_grad():
                metrics = compute_metrics(logits.float(), targets)
            running_metrics["fgIoU"] += metrics["fgIoU"]
            running_metrics["mIoU"]  += metrics["mIoU"]

            pbar.update(batch_episodes)

            # ── Validation ──
            if episode_count - last_val_episode >= cfg.VAL_INTERVAL:
                last_val_episode = episode_count
                elapsed = time.time() - start_time

                train_loss  = running_loss / running_count
                train_miou  = running_metrics["mIoU"] / running_count
                train_fgiou = running_metrics["fgIoU"] / running_count
                running_loss    = 0
                running_metrics = {"fgIoU": 0, "mIoU": 0}
                running_count   = 0

                val_loss, val_metrics = validate(model, val_loader, focal_criterion, device)
                model.train()  # BN override: frozen katmanlar eval'da kalır

                current_lr = optimizer.param_groups[0]["lr"]

                mlflow.log_metrics({
                    "train_loss":      train_loss,
                    "train_mIoU":      train_miou,
                    "train_fgIoU":     train_fgiou,
                    "val_loss":        val_loss,
                    "val_mIoU":        val_metrics["mIoU"],
                    "val_fgIoU":       val_metrics["fgIoU"],
                    "val_precision":   val_metrics["precision"],
                    "val_recall":      val_metrics["recall"],
                    "val_f1":          val_metrics["f1"],
                    "lr":              current_lr,
                }, step=episode_count)

                pbar.set_postfix({
                    "t_mIoU":  f"{train_miou:.3f}",
                    "v_mIoU":  f"{val_metrics['mIoU']:.3f}",
                    "v_fgIoU": f"{val_metrics['fgIoU']:.3f}",
                    "lr":      f"{current_lr:.1e}",
                })

                tqdm.write(
                    f"  [{episode_count:6d}/{cfg.TOTAL_EPISODES}] "
                    f"({elapsed/60:.1f}m) "
                    f"train_loss={train_loss:.4f} mIoU={train_miou:.3f} | "
                    f"val_loss={val_loss:.4f} mIoU={val_metrics['mIoU']:.3f} "
                    f"fgIoU={val_metrics['fgIoU']:.3f}"
                )

                # Best model kaydı
                if val_metrics["mIoU"] > best_val_miou:
                    best_val_miou = val_metrics["mIoU"]
                    best_episode  = episode_count

                    ckpt_dir  = Path(cfg.EXPERIMENTS_DIR) / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"best_{run_name}.pt"

                    torch.save({
                        "episode":            episode_count,
                        "model_state_dict":   model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_mIoU":           best_val_miou,
                        "val_fgIoU":          val_metrics["fgIoU"],
                        "config":             cfg.model_dump(),
                    }, ckpt_path)

                    tqdm.write(f"  → Best model (mIoU={best_val_miou:.4f})")

        pbar.close()

        total_time = time.time() - start_time
        mlflow.log_metric("best_val_mIoU", best_val_miou)
        mlflow.log_metric("best_episode", best_episode)
        mlflow.log_metric("total_episodes_trained", episode_count)
        mlflow.log_metric("total_time_minutes", total_time / 60)

        print(f"\n  Run:          {run_name}")
        print(f"  Best mIoU:    {best_val_miou:.4f} @ episode {best_episode}")
        print(f"  Time:         {total_time/60:.1f} min")

    return best_val_miou


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    folds_to_run = list(range(cfg.KFOLD_N_SPLITS))

    seed   = cfg.RANDOM_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name()}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)

    results = []
    for fold_i in folds_to_run:
        setup_seed(seed)
        best_miou = run_fold(fold_i, seed, device)
        results.append(best_miou)

    if len(results) > 1:
        arr = np.array(results)
        print(f"\n{'='*60}")
        print(f"  5-Fold Özet — {BACKBONE_LABELS[cfg.BACKBONE]} / {PRETRAINED_LABELS[cfg.PRETRAINED]} / {cfg.UNFREEZE_FROM}")
        print(f"  Val mIoU: {arr.mean():.4f} ± {arr.std():.4f}")
        for i, v in enumerate(results):
            print(f"    fold{i}: {v:.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
