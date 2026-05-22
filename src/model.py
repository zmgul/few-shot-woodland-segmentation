"""
model.py — ProtoNet for Few-Shot Semantic Segmentation
========================================================
Desteklenen backbone'lar:
  CNN:         ResNet-18/34/50/101/152, Xception
  Transformer: Swin-T, ViT-B/16

Kanal çıkışları:
  ResNet-18/34 (BasicBlock):  512 → feature_reduce → 256
  ResNet-50/101/152 (Bottleneck): 2048 → feature_reduce → 256
  Xception:                   2048 → feature_reduce → 256
  Swin-T:                      768 → feature_reduce → 256
  ViT-B/16:                    768 → feature_reduce → 256

Dilated stride=8 (CNN backbones):
  Bottleneck: conv2.stride=1 + dilation uygulanır
  BasicBlock: conv1.stride=1 kaldırılır, conv2'ye dilation eklenir

Unfreeze desteği:
  "none"   → Tüm backbone frozen
  "layer4" → ResNet: layer4 açık | Diğer: NotImplementedError (Phase 3'e bırakıldı)
  "layer3" → ResNet: layer3+4 açık | Diğer: NotImplementedError

BN Override:
  Frozen backbone katmanlarındaki BN'ler model.train() çağrısında da eval
  modunda kalır — ImageNet running_mean/running_var korunur.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Backbone meta bilgileri ──────────────────────────────────────────────────

BACKBONE_CHANNELS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "xception": 2048,
    "swin_t": 768,
    "vit_b16": 768,
}

# MLflow run_name ve checkpoint adında kullanılan etiketler
BACKBONE_LABELS = {
    "resnet18":  "ResNet18",
    "resnet34":  "ResNet34",
    "resnet50":  "ResNet50",
    "resnet101": "ResNet101",
    "resnet152": "ResNet152",
    "xception":  "Xception",
    "swin_t":    "SwinT",
    "vit_b16":   "ViTB16",
}

PRETRAINED_LABELS = {
    "imagenet_v1": "ImageNetV1",
    "million_aid": "MillionAID",
    "bigearthnet": "BigEarthNet",
    "seco":        "SeCo",
    "cityscapes":  "Cityscapes",
}

# Harici checkpoint dosya adları (imagenet_v1 hariç — torchvision'dan gelir)
PRETRAINED_CKPT_FILES = {
    "million_aid": "million_aid_resnet50.pth",
    "bigearthnet": "bigearthnet_resnet50.safetensors",
    "seco":        "seco_resnet50.pth",
    "cityscapes":  "cityscapes_resnet50.pth",
}


class ProtoNet(nn.Module):

    # ResNet: layer3, layer4
    # Swin-T: stage3 (features[4-5]), stage4 (features[6-7])
    UNFREEZE_OPTIONS = {
        "none":   [],
        "layer4": ["layer4"],
        "layer3": ["layer3", "layer4"],
        "stage4": ["swin_stage4"],
        "stage3": ["swin_stage3", "swin_stage4"],
    }

    def __init__(self, backbone_name="resnet50", feature_dim=256,
                 initial_temperature=10.0, unfreeze_from="none",
                 input_size=473, pretrained="imagenet_v1",
                 pretrained_ckpt_dir=None):
        """
        Args:
            backbone_name:       Backbone mimarisi (bkz. BACKBONE_CHANNELS)
            feature_dim:         feature_reduce çıkış boyutu
            initial_temperature: Öğrenilen sıcaklık parametresinin başlangıcı
            unfreeze_from:       Açılacak backbone katmanı ("none"/"layer4"/"layer3")
            input_size:          Giriş görüntü boyutu (ViT pos_embed için gerekli)
            pretrained:          Ön-eğitimli ağırlık kaynağı (bkz. PRETRAINED_LABELS)
            pretrained_ckpt_dir: Harici checkpoint dizini (imagenet_v1 hariç)
        """
        super().__init__()

        if backbone_name not in BACKBONE_CHANNELS:
            raise ValueError(
                f"Desteklenmeyen backbone: '{backbone_name}'. "
                f"Seçenekler: {list(BACKBONE_CHANNELS.keys())}"
            )
        if unfreeze_from not in self.UNFREEZE_OPTIONS:
            raise ValueError(
                f"unfreeze_from='{unfreeze_from}' geçersiz. "
                f"Seçenekler: {list(self.UNFREEZE_OPTIONS.keys())}"
            )
        if pretrained not in PRETRAINED_LABELS:
            raise ValueError(
                f"pretrained='{pretrained}' geçersiz. "
                f"Seçenekler: {list(PRETRAINED_LABELS.keys())}"
            )

        self.backbone_name = backbone_name
        self.feature_dim = feature_dim
        self.unfreeze_from = unfreeze_from
        self.input_size = input_size
        self.pretrained = pretrained

        # ── Backbone ──
        self._build_backbone(backbone_name, input_size, pretrained, pretrained_ckpt_dir)
        self._setup_unfreeze(unfreeze_from)

        # ── Eğitilebilir katmanlar ──
        in_channels = BACKBONE_CHANNELS[backbone_name]
        self.feature_reduce = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(initial_temperature))
        )

        self._print_param_report()

    # ─────────────────────────────────────────────────────────────────────────
    #  Backbone kurulumu
    # ─────────────────────────────────────────────────────────────────────────

    def _build_backbone(self, name, input_size, pretrained, pretrained_ckpt_dir):
        if name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
            self._build_resnet(name, pretrained, pretrained_ckpt_dir)
        elif name == "xception":
            self._build_xception()
        elif name == "swin_t":
            self._build_swin_t()
        elif name == "vit_b16":
            self._build_vit_b16(input_size)

    def _build_resnet(self, name, pretrained, pretrained_ckpt_dir):
        from torchvision.models import (
            resnet18, resnet34, resnet50, resnet101, resnet152,
            ResNet18_Weights, ResNet34_Weights,
            ResNet50_Weights, ResNet101_Weights, ResNet152_Weights,
        )
        model_fn_map = {
            "resnet18":  (resnet18,  ResNet18_Weights.IMAGENET1K_V1),
            "resnet34":  (resnet34,  ResNet34_Weights.IMAGENET1K_V1),
            "resnet50":  (resnet50,  ResNet50_Weights.IMAGENET1K_V1),
            "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
            "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V1),
        }
        model_fn, imagenet_weights = model_fn_map[name]

        if pretrained == "imagenet_v1":
            backbone = model_fn(weights=imagenet_weights)
        else:
            # Harici pretrained: önce boş model, sonra checkpoint yükle
            backbone = model_fn(weights=None)
            self._load_external_pretrained(backbone, pretrained, pretrained_ckpt_dir)

        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Dilated stride=32 → stride=8
        self._make_dilated(self.layer3, dilation=2)
        self._make_dilated(self.layer4, dilation=4)

        self._backbone_type = "resnet"
        self._backbone_layer_names = ["layer0", "layer1", "layer2", "layer3", "layer4"]

    def _load_external_pretrained(self, backbone, pretrained, ckpt_dir):
        """Harici pretrained checkpoint'u ResNet backbone'a yükler."""
        if ckpt_dir is None:
            raise ValueError(
                f"pretrained='{pretrained}' için pretrained_ckpt_dir gerekli."
            )

        ckpt_path = Path(ckpt_dir) / PRETRAINED_CKPT_FILES[pretrained]
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint bulunamadı: {ckpt_path}\n"
                f"İndir ve {ckpt_dir}/ altına koy."
            )

        # safetensors veya torch formatını yükle
        if ckpt_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Yaygın sarmalayıcı anahtarları çöz
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

        # Yaygın prefix'leri temizle (en uzun prefix önce denenecek şekilde sıralı)
        cleaned = {}
        prefixes = (
            "model.vision_encoder.",   # BIFOLD BigEarthNet
            "backbone.", "encoder.", "module.", "resnet.", "model.",
        )
        for k, v in state_dict.items():
            for prefix in prefixes:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v

        # Multi-band conv1 → RGB
        # BigEarthNet S2: 10 band (B02,B03,B04,...) → RGB = [2,1,0] (Red,Green,Blue)
        # SeCo/diğer S2: benzer band sıralaması
        conv1_key = "conv1.weight"
        if conv1_key in cleaned and cleaned[conv1_key].shape[1] != 3:
            n_channels = cleaned[conv1_key].shape[1]
            # Sentinel-2 band sırası: B02(Blue=0), B03(Green=1), B04(Red=2)
            # RGB sırası: R=2, G=1, B=0
            rgb_indices = [2, 1, 0]
            print(f"  [PRETRAINED] {pretrained}: conv1 {n_channels}-band → RGB (B04,B03,B02)")
            cleaned[conv1_key] = cleaned[conv1_key][:, rgb_indices, :, :]

        # fc katmanını çıkar (sınıf sayısı farklı olabilir)
        cleaned = {k: v for k, v in cleaned.items() if not k.startswith("fc.")}

        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        loaded = len(cleaned) - len(unexpected)
        print(f"  [PRETRAINED] {pretrained}: {loaded} parametre yüklendi")
        if missing:
            print(f"  [PRETRAINED] Eksik anahtarlar: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  [PRETRAINED] Fazla anahtarlar: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    def _build_xception(self):
        import timm
        # forward_features → [B, 2048, H/32, W/32]
        self.xception_backbone = timm.create_model("xception", pretrained=True)
        self._backbone_type = "xception"
        self._backbone_layer_names = ["xception_backbone"]

    def _build_swin_t(self):
        from torchvision.models import swin_t, Swin_T_Weights
        backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        # torchvision Swin-T features: [0-1]=stage1, [2-3]=stage2, [4-5]=stage3, [6-7]=stage4
        self.swin_stem = nn.Sequential(*[backbone.features[i] for i in range(4)])   # stage1+2
        self.swin_stage3 = nn.Sequential(*[backbone.features[i] for i in range(4, 6)])  # patch_merge + blocks
        self.swin_stage4 = nn.Sequential(*[backbone.features[i] for i in range(6, 8)])  # patch_merge + blocks
        self.swin_norm = backbone.norm
        self._backbone_type = "swin_t"
        self._backbone_layer_names = ["swin_stem", "swin_stage3", "swin_stage4", "swin_norm"]

    def _build_vit_b16(self, input_size):
        import timm
        # img_size belirtmek pretrained pos_embed'i input_size için interpole eder
        self.vit_backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            img_size=input_size,
        )
        self._backbone_type = "vit_b16"
        self._backbone_layer_names = ["vit_backbone"]

    # ─────────────────────────────────────────────────────────────────────────
    #  Dondurma / çözme
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_unfreeze(self, unfreeze_from):
        # Tüm backbone parametrelerini dondur
        for layer_name in self._backbone_layer_names:
            for param in getattr(self, layer_name).parameters():
                param.requires_grad = False

        layers_to_unfreeze = self.UNFREEZE_OPTIONS[unfreeze_from]

        if layers_to_unfreeze and self._backbone_type not in ("resnet", "swin_t"):
            raise NotImplementedError(
                f"unfreeze_from='{unfreeze_from}', backbone='{self.backbone_name}': "
                f"Bu backbone için katman bazlı unfreeze desteklenmiyor. "
                f"Sadece 'none' kullanılabilir."
            )

        # ResNet: belirtilen katmanları aç
        for layer_name in layers_to_unfreeze:
            for param in getattr(self, layer_name).parameters():
                param.requires_grad = True

        # Frozen layer listesi (train() override için)
        all_layers = set(self._backbone_layer_names)
        self._frozen_layers = all_layers - set(layers_to_unfreeze)

    # ─────────────────────────────────────────────────────────────────────────
    #  BN Override: frozen katmanlar eval modunda kalır
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, mode=True):
        """
        Frozen backbone katmanlarının BN'lerini eval'da tutar.
        ImageNet running_mean/running_var'ın bozulmasını önler.
        """
        super().train(mode)
        if mode:
            for layer_name in self._frozen_layers:
                layer = getattr(self, layer_name, None)
                if layer is not None:
                    layer.eval()
        return self

    # ─────────────────────────────────────────────────────────────────────────
    #  Dilated convolution (CNN backbones)
    # ─────────────────────────────────────────────────────────────────────────

    def _make_dilated(self, layer, dilation):
        for block in layer:
            # Downsample path: stride=1
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)

            if hasattr(block, "conv3"):
                # Bottleneck: stride conv2'de (3×3 conv)
                block.conv2.stride = (1, 1)
                block.conv2.dilation = (dilation, dilation)
                block.conv2.padding = (dilation, dilation)
            else:
                # BasicBlock: stride conv1'de, dilation conv2'ye
                block.conv1.stride = (1, 1)
                block.conv2.dilation = (dilation, dilation)
                block.conv2.padding = (dilation, dilation)

    # ─────────────────────────────────────────────────────────────────────────
    #  Feature çıkarma
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features(self, images):
        if self._backbone_type == "resnet":
            return self._extract_resnet(images)
        elif self._backbone_type == "xception":
            return self._extract_xception(images)
        elif self._backbone_type == "swin_t":
            return self._extract_swin_t(images)
        elif self._backbone_type == "vit_b16":
            return self._extract_vit_b16(images)

    def _extract_resnet(self, images):
        unfrozen_set = set(self.UNFREEZE_OPTIONS[self.unfreeze_from])
        x = images
        for layer_name in ["layer0", "layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self, layer_name)
            if layer_name in unfrozen_set:
                x = layer(x)
            else:
                with torch.no_grad():
                    x = layer(x)
        return self.feature_reduce(x)

    def _extract_xception(self, images):
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            x = self.xception_backbone.forward_features(images.float())
        return self.feature_reduce(x)

    def _extract_swin_t(self, images):
        unfrozen_set = set(self.UNFREEZE_OPTIONS[self.unfreeze_from])
        # Stem (stage1+2): always frozen
        with torch.no_grad():
            x = self.swin_stem(images)
        # Stage3
        if "swin_stage3" in unfrozen_set:
            x = self.swin_stage3(x)
        else:
            with torch.no_grad():
                x = self.swin_stage3(x)
        # Stage4
        if "swin_stage4" in unfrozen_set:
            x = self.swin_stage4(x)
        else:
            with torch.no_grad():
                x = self.swin_stage4(x)
        # Norm
        with torch.no_grad():
            x = self.swin_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, 768, H', W']
        return self.feature_reduce(x)

    def _extract_vit_b16(self, images):
        with torch.no_grad():
            # forward_features: [B, 1+N, 768] (class token + patch token'lar)
            x = self.vit_backbone.forward_features(images)
        # Prefix token'ları çıkar (genelde 1 class token)
        n_prefix = getattr(self.vit_backbone, "num_prefix_tokens", 1)
        x = x[:, n_prefix:]          # [B, N, 768]
        B, N, C = x.shape
        h = w = int(N ** 0.5)
        if h * w != N:
            raise RuntimeError(
                f"ViT: N={N} kare değil (h*w={h*w}). "
                f"input_size={self.input_size} patch_size=16 ile "
                f"(input_size//16)^2 = {(self.input_size//16)**2} bekleniyor."
            )
        x = x.permute(0, 2, 1).reshape(B, C, h, w).contiguous()  # [B, 768, h, w]
        return self.feature_reduce(x)

    # ─────────────────────────────────────────────────────────────────────────
    #  ProtoNet forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, support_images, support_masks, query_image):
        """
        Args:
            support_images: [B, K, 3, H, W]
            support_masks:  [B, K, H, W]
            query_image:    [B, 3, H, W]
        Returns:
            logits: [B, 1, H, W]
        """
        batch_size, k_shot = support_images.shape[:2]
        input_height, input_width = query_image.shape[2:]

        # Support features
        support_flat = support_images.view(
            batch_size * k_shot, 3, input_height, input_width
        )
        support_features = self.extract_features(support_flat)
        feature_height, feature_width = support_features.shape[2:]
        support_features = support_features.view(
            batch_size, k_shot, self.feature_dim, feature_height, feature_width
        )

        # Maskeleri feature boyutuna küçült
        masks_flat = support_masks.float().view(
            batch_size * k_shot, 1,
            support_masks.shape[2], support_masks.shape[3]
        )
        masks_resized = F.interpolate(
            masks_flat, size=(feature_height, feature_width), mode="nearest"
        ).view(batch_size, k_shot, 1, feature_height, feature_width)

        # FG + BG prototipler
        fg_prototype = self._masked_average_pooling(support_features, masks_resized)
        bg_prototype = self._masked_average_pooling(support_features, 1.0 - masks_resized)

        fg_prototype = F.normalize(fg_prototype, dim=1)
        bg_prototype = F.normalize(bg_prototype, dim=1)

        # Query segmentasyon
        query_features = self.extract_features(query_image)
        query_features = F.normalize(query_features, dim=1)

        fg_sim = (query_features * fg_prototype.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        bg_sim = (query_features * bg_prototype.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)

        temperature = torch.exp(self.log_temperature)
        logits = temperature * (fg_sim - bg_sim)

        logits = F.interpolate(
            logits, size=(input_height, input_width),
            mode="bilinear", align_corners=False
        )
        return logits

    def _masked_average_pooling(self, features, masks, epsilon=1e-6):
        masked = features * masks
        feature_sum = masked.sum(dim=(1, 3, 4))
        mask_sum = masks.sum(dim=(1, 3, 4))
        return feature_sum / (mask_sum + epsilon)

    def _print_param_report(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f"\n[MODEL] ProtoNet — backbone={self.backbone_name}, pretrained={self.pretrained}, unfreeze={self.unfreeze_from}")
        print(f"  Eğitilebilir: {trainable:>12,} parametre")
        print(f"  Frozen:       {frozen:>12,} parametre")
        print(f"  Toplam:       {total:>12,} parametre")

    @torch.no_grad()
    def predict(self, support_images, support_masks, query_image):
        logits = self.forward(support_images, support_masks, query_image)
        return (logits.squeeze(1) > 0).long()
