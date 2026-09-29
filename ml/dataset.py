"""PyTorch Dataset for crop prediction training."""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from ml.preprocessing import AspectRatioResize, get_train_transform, get_val_transform


class CropDataset(Dataset):
    """
    Dataset for crop prediction.
    
    Each sample is:
        - An image (resized with aspect-ratio preservation + padding)
        - A crop label [x, y, width, height] (normalized 0–1)
    """

    def __init__(
        self,
        examples: List[dict],
        image_size: int = 224,
        is_training: bool = True,
    ):
        """
        Args:
            examples: List of dicts with keys:
                - original_path: str
                - norm_x, norm_y, norm_width, norm_height: float
            image_size: Target image size for the model
            is_training: Whether to apply training augmentation
        """
        self.examples = examples
        self.image_size = image_size
        self.resizer = AspectRatioResize(target_size=image_size)
        self.transform = get_train_transform(image_size) if is_training else get_val_transform(image_size)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[idx]

        # Load image
        img = Image.open(example["original_path"]).convert("RGB")

        # Resize with aspect ratio preservation
        padded_img, metadata = self.resizer(img)

        # Apply transforms (color jitter for training, just normalize for val)
        tensor = self.transform(padded_img)

        # Crop label — these are already normalized to [0, 1] relative to the original
        label = torch.tensor([
            example["norm_x"],
            example["norm_y"],
            example["norm_width"],
            example["norm_height"],
        ], dtype=torch.float32)

        return tensor, label


def build_dataloaders(
    train_examples: List[dict],
    val_examples: List[dict],
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and validation DataLoaders."""
    train_dataset = CropDataset(train_examples, image_size=image_size, is_training=True)
    val_dataset = CropDataset(val_examples, image_size=image_size, is_training=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't benefit from pin_memory
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader
