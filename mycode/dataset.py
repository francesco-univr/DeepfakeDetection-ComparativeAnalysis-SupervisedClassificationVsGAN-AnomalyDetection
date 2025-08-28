import os
import glob
from typing import Optional, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Custom PyTorch dataset for DeepFake detection
def _build_transform(split: str, use_imagenet_stats: bool, size: int):
    # Normalization statistics
    if use_imagenet_stats:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# For the training set, apply random augmentations to improve model generalization
    if split == "train":
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.1, 0.1, 0.1, 0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    # For validation and test sets, use simple resizing and center cropping for consistent evaluation
    else:
        return T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


class DeepFakeDataset(Dataset):
    # Find the correct data split directory
    def __init__(
        self,
        root: str,
        split: str = "train",
        use_imagenet_stats: bool = False,
        size: int = 224,
        transform: Optional[torch.nn.Module] = None,
    ):
        assert split in {"train", "val", "validation", "test"}, "Split not valid"
        # Alias 'val' -> 'validation' if exists
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir) and split == "val":
            split_dir = os.path.join(root, "validation")
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Directory split non found: {split_dir}")

        # Group files
        self.samples: List[Tuple[str, int]] = []
        for label, cls in ((0, "Real"), (1, "Fake")):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for path in glob.glob(os.path.join(cls_dir, pattern)):
                    self.samples.append((path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}/{{Real,Fake}} "
                f"with extensions {VALID_EXTS}"
            )

        self.split = "validation" if split == "val" else split
        self.use_imagenet_stats = use_imagenet_stats
        self.size = size
        self.transform = transform if transform is not None else _build_transform(
            self.split, self.use_imagenet_stats, self.size
        )
# Return the total number of samples in the dataset
    def __len__(self) -> int:
        return len(self.samples)
    
# Load and return a single sample from the dataset
    def __getitem__(self, idx: int):
        path, lab = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), lab

