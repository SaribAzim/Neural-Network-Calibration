"""
model_loader.py
---------------
Loads a pretrained ResNet-50 and adapts its final layer for CIFAR-10
(10 classes instead of the default 1000 ImageNet classes).

Why ResNet-50?
--------------
It is a well-known, widely-reproduced model.
It is known to be overconfident on CIFAR-10 — which makes it a perfect
subject for a calibration study.
"""

import torch
import torch.nn as nn
from torchvision import models


def load_resnet50(
    num_classes: int = 10,
    pretrained: bool = True,
    device: torch.device = None,
) -> nn.Module:
    """
    Load ResNet-50 with ImageNet weights and replace the head for CIFAR-10.

    Architecture change:
        Original  → fc: Linear(2048, 1000)
        Ours      → fc: Linear(2048, 10)

    We also replace the 7x7 stem conv (designed for 224x224 images) with a
    3x3 conv better suited to CIFAR-10's 32x32 inputs, and remove the
    max-pooling layer — a standard adaptation for small images.

    Args:
        num_classes : number of output classes (10 for CIFAR-10)
        pretrained  : if True, load ImageNet weights for the backbone
        device      : torch device; defaults to CUDA if available

    Returns:
        model : nn.Module ready for training or inference
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model   = models.resnet50(weights=weights)

    # ── Adapt stem for 32×32 inputs ─────────────────────────────────────────
    # The default 7×7 conv with stride 2 + maxpool halves resolution twice,
    # leaving only 4×4 feature maps from a 32×32 input — too aggressive.
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()   # remove max-pool

    # ── Replace classification head ──────────────────────────────────────────
    in_features   = model.fc.in_features  # 2048
    model.fc      = nn.Linear(in_features, num_classes)

    model = model.to(device)
    print(f"[ModelLoader] ResNet-50 loaded → {num_classes} classes | device: {device}")
    return model