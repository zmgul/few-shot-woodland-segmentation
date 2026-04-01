"""
Few-shot Semantic Segmentation Utils
- Dice Loss
- Focal Loss
- Compute Metrics 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss - assigns more weight to hard examples
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Binary focal loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        return loss.mean(dim=(2, 3))


def dice_loss(logits, targets):
    """Dice Loss for segmentation"""
    probs = torch.sigmoid(logits)
    smooth = 1.0
    inter = (probs * targets).sum(dim=(2, 3))
    union = (probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)))
    return 1 - (2 * inter + smooth) / (union + smooth)


def compute_metrics(logits, targets, threshold=0.5):
    """
    Compute IoU, Precision, Recall, F1, and mIoU for Academic Reporting.
    Targets and Preds are expected in binary format.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    epsilon = 1e-7

    # Foreground Metrics 
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    # fgIoU 
    inter_fg = tp
    union_fg = tp + fp + fn
    iou_fg = (inter_fg + epsilon) / (union_fg + epsilon)

    # Precision, Recall, F1
    prec = (tp + epsilon) / (tp + fp + epsilon)
    rec = (tp + epsilon) / (tp + fn + epsilon)
    f1 = 2 * (prec * rec) / (prec + rec + epsilon)

    # Background Metrics For mIoU Calculation 
    # Background IoU: overlap accuracy of background pixels
    inter_bg = tn
    union_bg = tn + fn + fp
    iou_bg = (inter_bg + epsilon) / (union_bg + epsilon)
    miou = (iou_fg + iou_bg) / 2

    return {
        'fgIoU': iou_fg.item(),
        'mIoU': miou.item(),
        'precision': prec.item(),
        'recall': rec.item(),
        'f1': f1.item()
    }
