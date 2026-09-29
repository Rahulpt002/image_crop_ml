"""
CropNet — Pretrained backbone with regression head for crop prediction.

Architecture:
    Pretrained CNN backbone (default: MobileNetV3-Small)
    → Global Average Pooling
    → Regression head: FC → ReLU → Dropout → FC → Sigmoid
    → 4 outputs: [x, y, width, height] (normalized 0–1)

The backbone is modular and can be swapped. All backbones are loaded
from torchvision with pretrained ImageNet weights.
"""

import torch
import torch.nn as nn
from torchvision import models


BACKBONE_REGISTRY = {
    "mobilenet_v3_small": {
        "constructor": models.mobilenet_v3_small,
        "weights": models.MobileNet_V3_Small_Weights.DEFAULT,
        "feature_dim": 576,
    },
    "mobilenet_v3_large": {
        "constructor": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.DEFAULT,
        "feature_dim": 960,
    },
    "efficientnet_b0": {
        "constructor": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.DEFAULT,
        "feature_dim": 1280,
    },
    "resnet18": {
        "constructor": models.resnet18,
        "weights": models.ResNet18_Weights.DEFAULT,
        "feature_dim": 512,
    },
    "resnet34": {
        "constructor": models.resnet34,
        "weights": models.ResNet34_Weights.DEFAULT,
        "feature_dim": 512,
    },
}


class CropNet(nn.Module):
    """
    Neural network that predicts crop rectangles from images.
    
    Input: Image tensor [B, 3, H, W]
    Output: Normalized crop coords [B, 4] — (x, y, width, height)
    
    All outputs are in [0, 1] via sigmoid activation.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. "
                f"Available: {list(BACKBONE_REGISTRY.keys())}"
            )

        config = BACKBONE_REGISTRY[backbone_name]
        self.backbone_name = backbone_name
        self.feature_dim = config["feature_dim"]

        # Load pretrained backbone
        if pretrained:
            backbone = config["constructor"](weights=config["weights"])
        else:
            backbone = config["constructor"](weights=None)

        # Remove the classification head — keep only features
        if "resnet" in backbone_name:
            # ResNet: remove fc layer, keep avgpool
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.pool = nn.Flatten()
        elif "mobilenet" in backbone_name:
            self.features = backbone.features
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        elif "efficientnet" in backbone_name:
            self.features = backbone.features
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        else:
            self.features = backbone.features
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        # Regression head: predict [x, y, width, height]
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 4),
            nn.Sigmoid(),  # Constrain outputs to [0, 1]
        )

        # Initialize head weights
        self._init_head()

    def _init_head(self):
        """Initialize the regression head with reasonable defaults."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # Initialize bias to predict a centered 50% crop
                    nn.init.constant_(m.bias, 0.0)

        # Set the final layer bias to produce a reasonable initial crop
        # Sigmoid(0) = 0.5, so bias=0 → predicts (0.5, 0.5, 0.5, 0.5)
        # We want roughly: x=0.15, y=0.1, w=0.7, h=0.8
        final_layer = self.head[-2]  # Linear before Sigmoid
        with torch.no_grad():
            # logit(0.15)≈-1.73, logit(0.1)≈-2.20, logit(0.7)≈0.85, logit(0.8)≈1.39
            final_layer.bias.copy_(torch.tensor([-1.73, -2.20, 0.85, 1.39]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Image tensor [B, 3, H, W]
            
        Returns:
            Crop coordinates [B, 4] — (x, y, width, height), all in [0, 1]
        """
        features = self.features(x)
        pooled = self.pool(features)
        crop = self.head(pooled)
        return crop

    def predict_with_validation(
        self,
        x: torch.Tensor,
        aspect_ratio: float | None = None,
        image_aspect_ratio: float | None = None,
    ) -> torch.Tensor:
        """
        Predict with post-processing to ensure valid crop boxes.
        
        Ensures:
            x + width ≤ 1
            y + height ≤ 1
        
        If aspect_ratio is provided, constrains the crop to that ratio (w/h).
        If image_aspect_ratio (image_width / image_height) is provided,
        converts the physical target aspect ratio to normalized coordinates.
        """
        crop = self.forward(x)
        
        x_coord = crop[:, 0]
        y_coord = crop[:, 1]
        width = crop[:, 2]
        height = crop[:, 3]

        # Clamp so crop doesn't exceed boundaries
        width = torch.clamp(width, max=1.0 - x_coord)
        height = torch.clamp(height, max=1.0 - y_coord)

        result = torch.stack([x_coord, y_coord, width, height], dim=1)

        if aspect_ratio is not None:
            result = constrain_to_aspect_ratio(
                result,
                aspect_ratio=aspect_ratio,
                image_aspect_ratio=image_aspect_ratio,
            )

        return result


def constrain_to_aspect_ratio(
    pred: torch.Tensor,
    aspect_ratio: float = 0.75,
    image_aspect_ratio: float | None = None,
) -> torch.Tensor:
    """
    Constrain predicted crop boxes to a target aspect ratio.

    Adjusts the crop dimensions so that width/height == aspect_ratio,
    re-centering within the original predicted area while keeping
    the box inside [0, 1].

    Args:
        pred: Tensor [B, 4] with (x, y, width, height) in [0, 1]
        aspect_ratio: Target physical crop width/height ratio (default 0.75 = 3:4)
        image_aspect_ratio: Optional original image width/height ratio.
            If provided, effective normalized AR = aspect_ratio / image_aspect_ratio.

    Returns:
        Tensor [B, 4] with constrained crop boxes
    """
    effective_ar = aspect_ratio if image_aspect_ratio is None else (aspect_ratio / image_aspect_ratio)

    x = pred[:, 0]
    y = pred[:, 1]
    w = pred[:, 2]
    h = pred[:, 3]

    # Two options: shrink width to match height, or shrink height to match width
    w_from_h = h * effective_ar       # width if we keep height
    h_from_w = w / effective_ar       # height if we keep width

    # Pick the option that produces a smaller (safer) crop
    use_height_based = w_from_h <= w   # can shrink width to fit?

    new_w = torch.where(use_height_based, w_from_h, w)
    new_h = torch.where(use_height_based, h, h_from_w)

    # Re-center relative to original crop center
    cx = x + w / 2
    cy = y + h / 2
    new_x = cx - new_w / 2
    new_y = cy - new_h / 2

    # Clamp to [0, 1] bounds
    new_x = torch.clamp(new_x, min=0.0)
    new_y = torch.clamp(new_y, min=0.0)
    new_x = torch.clamp(new_x, max=1.0 - new_w)
    new_y = torch.clamp(new_y, max=1.0 - new_h)

    # Safety: ensure w and h are positive
    new_w = torch.clamp(new_w, min=0.01)
    new_h = torch.clamp(new_h, min=0.01)

    return torch.stack([new_x, new_y, new_w, new_h], dim=1)


def get_device() -> torch.device:
    """Get the best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def build_model(backbone: str = "mobilenet_v3_small", pretrained: bool = True) -> CropNet:
    """Factory function to build a CropNet model."""
    return CropNet(backbone_name=backbone, pretrained=pretrained)

