# Last updated: 2026-05-09 | backbone selector
import torch
import torch.nn as nn
from torchvision import models

# ─────────────────────────────────────────────────────────────
# Registry: name → (constructor, weights, out_dim, family)
# ─────────────────────────────────────────────────────────────

BACKBONE_REGISTRY = {
    "resnet18": (
        models.resnet18, models.ResNet18_Weights.DEFAULT, 512, "resnet"
    ),
    "mobilenet_v3_small": (
        models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, 576, "mobilenet"
    ),
    "efficientnet_b0": (
        models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280, "efficientnet"
    ),
    "efficientnet_b2": (
        models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT, 1408, "efficientnet"
    ),
}

BACKBONE_NAMES = list(BACKBONE_REGISTRY.keys())


class GenericBackbone(nn.Module):
    """
    Multi-architecture feature extractor with selective layer unfreezing.

    Supports ResNet-18, MobileNetV3-Small, and EfficientNet-B0/B2.
    The final classification layer is removed; output is a flat feature
    vector of size out_dim. unfreeze_layers controls how many of the last
    blocks receive gradients.
    Use trainable_params() to pass unfrozen parameters to the optimizer
    with a separate (smaller) learning rate.
    """

    def __init__(self, backbone_name="resnet18", pretrained=True, unfreeze_layers=0):
        super().__init__()
        fn, weights, out_dim, family = BACKBONE_REGISTRY[backbone_name]
        model = fn(weights=weights if pretrained else None)

        self.backbone_name   = backbone_name
        self.out_dim         = out_dim
        self.family          = family
        self.unfreeze_layers = unfreeze_layers

        if family == "resnet":
            self.stem   = nn.Sequential(model.conv1, model.bn1,
                                        model.relu,  model.maxpool)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.pool   = model.avgpool
            _blocks = [self.layer1, self.layer2, self.layer3, self.layer4]

        elif family == "mobilenet":
            self.features = model.features
            self.pool     = model.avgpool
            _blocks = list(model.features.children())

        elif family == "efficientnet":
            self.features = model.features
            self.pool     = model.avgpool
            _blocks = list(model.features.children())

        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        # Selectively unfreeze the last N blocks
        self._unfrozen = _blocks[-unfreeze_layers:] if unfreeze_layers > 0 else []
        for block in self._unfrozen:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x):
        if self.family == "resnet":
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        elif self.family in ("mobilenet", "efficientnet"):
            x = self.features(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def get_backbone(device, backbone_name="resnet18", pretrained=True, unfreeze_layers=0):
    model = GenericBackbone(backbone_name, pretrained, unfreeze_layers)
    model = model.to(device)
    model.eval()
    return model
