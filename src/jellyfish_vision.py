#!/usr/bin/env python3
"""Training pipeline for jellyfish classification."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from albumentations import (
    Blur,
    CLAHE,
    GaussNoise,
    GridDistortion,
    IAAAdditiveGaussianNoise,
    IAAEmboss,
    IAAPiecewiseAffine,
    IAASharpen,
    MotionBlur,
    MedianBlur,
    OpticalDistortion,
    RandomBrightnessContrast,
    RandomRotate90,
    Flip,
    Transpose,
    ShiftScaleRotate,
    HueSaturationValue,
    Normalize,
    OneOf,
    Compose,
)
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_metric_learning import losses, miners
from sklearn.model_selection import train_test_split
from timm import create_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader


class CBAM(nn.Module):
    """Convolutional block attention module."""

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.channel_attention(x)
        x = x * ca
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa


class CustomEfficientNetB7(nn.Module):
    """EfficientNet-B7 backbone enhanced with CBAM and domain branches."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = create_model("efficientnet_b7", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.cbam = CBAM(feat_dim)
        self.underwater_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.beach_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.attn = nn.Sequential(nn.Conv2d(3, 1, 1), nn.Sigmoid())
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attn(x)
        u = self.underwater_conv(x * w)
        b = self.beach_conv(x * (1 - w))
        x = u + b
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.backbone.global_pool(x).flatten(1)
        return self.classifier(x)


class JellyfishDataset(Dataset):
    """Dataset that loads images via OpenCV and applies Albumentations."""

    def __init__(self, paths: List[Path], labels: List[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image {img_path} not found")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx]


class JellyfishModel(pl.LightningModule):
    """LightningModule encapsulating training and validation logic."""

    def __init__(self, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = CustomEfficientNetB7(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.metric_loss = losses.ArcFaceLoss(num_classes, self.model.backbone.num_features)
        self.miner = miners.MultiSimilarityMiner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        ml = self.metric_loss(logits, y, self.miner(logits, y))
        loss = self.criterion(logits, y) + ml
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr)
        sch = CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


def underwater_effect(img: np.ndarray) -> np.ndarray:
    blue = cv2.GaussianBlur(img[:, :, 0], (5, 5), 0)
    img[:, :, 0] = blue
    return img


def beach_effect(img: np.ndarray) -> np.ndarray:
    brightness = np.random.uniform(1.0, 1.5)
    contrast = np.random.uniform(0.8, 1.2)
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


class UnderwaterEffect(A.ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return underwater_effect(img)


class BeachEffect(A.ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return beach_effect(img)


def get_transforms(train: bool = True):
    if train:
        return Compose(
            [
                RandomRotate90(),
                Flip(),
                Transpose(),
                UnderwaterEffect(p=0.5),
                BeachEffect(p=0.5),
                OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.2),
                OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                ShiftScaleRotate(0.0625, 0.2, 45, p=0.2),
                OneOf(
                    [
                        OpticalDistortion(p=0.3),
                        GridDistortion(p=0.1),
                        IAAPiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),
                OneOf(
                    [
                        CLAHE(clip_limit=2),
                        IAASharpen(),
                        IAAEmboss(),
                        RandomBrightnessContrast(),
                    ],
                    p=0.3,
                ),
                HueSaturationValue(p=0.3),
                Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        return Compose([Normalize(), ToTensorV2()])


def parse_args():
    parser = argparse.ArgumentParser(description="JellyfishVision Trainer")
    parser.add_argument("--data-dir", type=Path, required=True, help="Root directory with images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def prepare_datasets(data_dir: Path):
    paths = list(data_dir.glob("**/*.jpg")) + list(data_dir.glob("**/*.png"))
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
    labels = [p.parent.name for p in paths]
    mapping = {l: i for i, l in enumerate(sorted(set(labels)))}
    lbls = [mapping[l] for l in labels]
    trv_p, te_p, trv_l, te_l = train_test_split(
        paths, lbls, test_size=0.15, stratify=lbls, random_state=42
    )
    tr_p, va_p, tr_l, va_l = train_test_split(
        trv_p, trv_l, test_size=0.1765, stratify=trv_l, random_state=42
    )
    ds_tr = JellyfishDataset(tr_p, tr_l, transform=get_transforms(True))
    ds_va = JellyfishDataset(va_p, va_l, transform=get_transforms(False))
    ds_te = JellyfishDataset(te_p, te_l, transform=get_transforms(False))
    return ds_tr, ds_va, ds_te, len(mapping)


def main():
    args = parse_args()
    ds_tr, ds_va, ds_te, n_classes = prepare_datasets(args.data_dir)
    num_workers = min(os.cpu_count() or 1, 4)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = JellyfishModel(num_classes=n_classes, lr=args.lr)
    ckpt = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="jelly-{epoch:02d}-{val_loss:.2f}",
    )
    es = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        callbacks=[es, ckpt],
        enable_progress_bar=True,
    )

    trainer.fit(model, dl_tr, dl_va)
    trainer.test(model, dl_te)
    torch.save(model.state_dict(), "final_model.pth")


if __name__ == "__main__":
    main()
