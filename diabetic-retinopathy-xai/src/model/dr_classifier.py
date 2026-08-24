"""
dr_classifier.py
----------------
Diabetic Retinopathy CNN classifier built on a pretrained EfficientNet-B3
backbone (via torchvision), with a custom classification head.

Architecture:
    EfficientNet-B3 (pretrained on ImageNet)
        └── Adaptive Average Pool
        └── Dropout(0.4)
        └── Linear(1536 → 512)
        └── BatchNorm1d(512)
        └── ReLU
        └── Dropout(0.3)
        └── Linear(512 → num_classes)   ← 5 DR severity grades

DR Severity Classes (0–4):
    0 — No DR
    1 — Mild
    2 — Moderate
    3 — Severe
    4 — Proliferative DR

Usage:
    from src.model.dr_classifier import DRClassifier, build_model
    model = build_model(num_classes=5, pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


# ── Constants ────────────────────────────────────────────────────────────────

NUM_CLASSES = 5
DR_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# EfficientNet-B3 output features before classification head
EFFICIENTNET_B3_FEATURES = 1536


# ── Model Definition ─────────────────────────────────────────────────────────

class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy severity classifier using EfficientNet-B3 backbone.

    Args:
        num_classes:    Number of output classes (default: 5).
        pretrained:     Use ImageNet pretrained weights (default: True).
        dropout_rate:   Dropout rate before first FC layer (default: 0.4).
        freeze_backbone: Freeze backbone weights for feature extraction mode.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        freeze_backbone: bool = False,
    ):
        super(DRClassifier, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # ── Backbone ─────────────────────────────────────────────────────────
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # Remove the original classifier head, keep only the feature extractor
        self.backbone = backbone.features          # Conv + MBConv blocks
        self.avg_pool = backbone.avgpool           # AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Custom Classification Head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(EFFICIENTNET_B3_FEATURES, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),      # lighter second dropout
            nn.Linear(512, num_classes),
        )

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Applies Kaiming normal initialization to linear layers."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.backbone(x)           # (B, 1536, H', W')
        pooled   = self.avg_pool(features)    # (B, 1536, 1, 1)
        flat     = torch.flatten(pooled, 1)   # (B, 1536)
        logits   = self.classifier(flat)      # (B, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> dict:
        """
        Returns predicted class index, name, and softmax probabilities.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Dict with keys: predicted_class, class_name, probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
            pred   = torch.argmax(probs, dim=1)

        return {
            "predicted_class": pred.cpu().numpy(),
            "class_names": [DR_CLASS_NAMES[i] for i in pred.cpu().numpy()],
            "probabilities": probs.cpu().numpy(),
        }

    def unfreeze_backbone(self, layers_from_end: int = 3) -> None:
        """
        Progressively unfreezes the last N blocks of the backbone
        for fine-tuning after initial feature-extraction training.

        Args:
            layers_from_end: Number of backbone blocks to unfreeze from end.
        """
        backbone_children = list(self.backbone.children())
        unfreeze_from = max(0, len(backbone_children) - layers_from_end)
        for i, child in enumerate(backbone_children):
            for param in child.parameters():
                param.requires_grad = i >= unfreeze_from

        unfrozen = sum(
            p.numel() for p in self.backbone.parameters()
            if p.requires_grad
        )
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"[UNFREEZE] Unfroze last {layers_from_end} backbone blocks: "
              f"{unfrozen:,} / {total:,} params trainable")

    def get_param_groups(self, backbone_lr: float = 1e-5,
                         head_lr: float = 1e-3) -> list:
        """
        Returns separate param groups for backbone and head
        (used with optimizers for differential learning rates).

        Args:
            backbone_lr: Learning rate for backbone parameters.
            head_lr:     Learning rate for classification head.

        Returns:
            List of param group dicts for torch.optim.
        """
        return [
            {"params": self.backbone.parameters(),   "lr": backbone_lr},
            {"params": self.avg_pool.parameters(),   "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]


# ── Factory Function ─────────────────────────────────────────────────────────

def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    device: str | None = None,
) -> DRClassifier:
    """
    Builds and returns a DRClassifier ready for training.

    Args:
        num_classes:     DR severity classes (default: 5).
        pretrained:      Use ImageNet pretrained backbone (default: True).
        dropout_rate:    Dropout rate (default: 0.4).
        freeze_backbone: Start with frozen backbone for transfer learning.
        device:          'cuda', 'cpu', or None (auto-detect).

    Returns:
        DRClassifier model moved to the target device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DRClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
    model = model.to(device)

    # Summary
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"\n{'='*55}")
    print(f"  DRClassifier — EfficientNet-B3 Backbone")
    print(f"{'='*55}")
    print(f"  Device          : {device}")
    print(f"  Output classes  : {num_classes}  {DR_CLASS_NAMES}")
    print(f"  Pretrained      : {pretrained}")
    print(f"  Backbone frozen : {freeze_backbone}")
    print(f"  Total params    : {total_params:>12,}")
    print(f"  Trainable params: {trainable_params:>12,}")
    print(f"{'='*55}\n")

    return model


# ── CLI Smoke Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running model smoke test...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(pretrained=False, freeze_backbone=False, device=device)

    # Dummy forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)

    assert output.shape == (4, NUM_CLASSES), \
        f"Expected (4, {NUM_CLASSES}), got {output.shape}"

    print(f"[OK] Forward pass — output shape: {output.shape}")
    print(f"[OK] Logits (first sample): {output[0].detach().cpu().numpy()}")

    # Test predict
    result = model.predict(dummy_input)
    print(f"[OK] Predictions: {result['class_names']}")
    print(f"[OK] Probabilities shape: {result['probabilities'].shape}")

    # Test unfreeze
    model.unfreeze_backbone(layers_from_end=3)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Trainable after partial unfreeze: {trainable:,}")

    print("\n[PASS] All smoke tests passed!")
