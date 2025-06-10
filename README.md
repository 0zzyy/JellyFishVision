# JellyFishVision

A PyTorch-Lightning pipeline for jellyfish species classification with CBAM, custom augmentations (underwater/beach effects).

## Features

- Custom EfficientNet-B7 + CBAM backbone  
- Underwater & beach domain-specific augmentations via Albumentations  
- Metric learning loss (ArcFace) alongside cross-entropy  
- Early stopping & checkpointing via Lightning callbacks  
- Test-time evaluation and final model export  

## Setup

```bash
git clone https://github.com/yourusername/JellyFishVision.git
cd JellyFishVision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

##Usage
Place your images under a folder structure like:

/path/to/data/train/<class_name>/*.jpg
/path/to/data/val/<class_name>  *.jpg
/path/to/data/test/<class_name> *.jpg

Then run:

python src/jellyfish_vision.py \
  --data-dir /path/to/data \
  --batch-size 32 \
  --max-epochs 50 \
  --gpus 1

##Arguments
|           Flag | Description                                |    Default   |
| -------------: | ------------------------------------------ | :----------: |
|   `--data-dir` | Root folder containing train/val/test dirs | **Required** |
| `--batch-size` | Batch size                                 |      32      |
| `--max-epochs` | Number of epochs                           |      50      |
|       `--gpus` | GPUs to use (0 for CPU)                    |       1      |

Results (best checkpoint & final_model.pth) and logs will be saved in the project root.


---

### `src/jellyfish_vision.py`

```python
#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from albumentations import Compose, RandomRotate90, Flip, Transpose, OneOf, \
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, Blur, \
    ShiftScaleRotate, OpticalDistortion, GridDistortion, IAAPiecewiseAffine, \
    CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, HueSaturationValue, \
    Normalize
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_metric_learning import losses, miners
from sklearn.model_selection import train_test_split
from timm import create_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa

class CustomEfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = create_model('efficientnet_b7', pretrained=True, num_classes=0)
        feat = self.backbone.num_features
        self.cbam = CBAM(feat)
        self.underwater_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.beach_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.attn = nn.Sequential(nn.Conv2d(3,1,1), nn.Sigmoid())
        self.classifier = nn.Linear(feat, num_classes)

    def forward(self, x):
        w = self.attn(x)
        u = self.underwater_conv(x * w)
        b = self.beach_conv(x * (1 - w))
        x = u + b
        x = self.backbone.forward_features(x)
        x = self.cbam(x)
        x = self.backbone.global_pool(x).flatten(1)
        return self.classifier(x)

class JellyfishDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        if img is None:
            raise FileNotFoundError(f"{self.paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, self.labels[idx]

class JellyfishModel(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = CustomEfficientNetB7(num_classes)
        self.crit = nn.CrossEntropyLoss()
        self.metric_loss = losses.ArcFaceLoss(num_classes, self.net.backbone.num_features)
        self.miner = miners.MultiSimilarityMiner()

    def forward(self, x): return self.net(x)

    def training_step(self, batch, _):
        x,y = batch
        logits = self(x)
        ml = self.metric_loss(logits,y,self.miner(logits,y))
        loss = self.crit(logits,y) + ml
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, _):
        x,y = batch
        logits = self(x)
        loss = self.crit(logits,y)
        acc = (logits.argmax(1)==y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr)
        sch = CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

# ──────────────────────────────────────────────────────────────────────────────
def get_transforms(train=True):
    if train:
        return Compose([
            RandomRotate90(), Flip(), Transpose(),
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.2),
            OneOf([MotionBlur(p=0.2), MedianBlur(3,p=0.1), Blur(3,p=0.1)], p=0.2),
            ShiftScaleRotate(0.0625,0.2,45,p=0.2),
            OneOf([OpticalDistortion(p=0.3), GridDistortion(p=0.1), IAAPiecewiseAffine(p=0.3)], p=0.2),
            OneOf([CLAHE(), IAASharpen(), IAAEmboss(), RandomBrightnessContrast()], p=0.3),
            HueSaturationValue(p=0.3),
            Normalize(), ToTensorV2(),
        ])
    else:
        return Compose([Normalize(), ToTensorV2()])

def parse_args():
    p = argparse.ArgumentParser("JellyFishVision")
    p.add_argument('--data-dir',      type=Path, required=True)
    p.add_argument('--batch-size',    type=int,   default=32)
    p.add_argument('--max-epochs',    type=int,   default=50)
    p.add_argument('--gpus',          type=int,   default=1)
    return p.parse_args()

def main():
    args = parse_args()
    # collect image paths
    data = args.data_dir
    all_paths = list(data.glob('**/*.jpg')) + list(data.glob('**/*.png'))
    labels = [p.parent.name for p in all_paths]
    mapping = {l:i for i,l in enumerate(sorted(set(labels)))}
    lbls = [mapping[l] for l in labels]

    # split
    trv_p, te_p, trv_l, te_l = train_test_split(all_paths, lbls,
                                                 test_size=0.15, stratify=lbls, random_state=42)
    tr_p, va_p, tr_l, va_l = train_test_split(trv_p, trv_l,
                                               test_size=0.1765,
                                               stratify=trv_l, random_state=42)

    # data loaders
    ds_tr = JellyfishDataset(tr_p, tr_l, transform=get_transforms(True))
    ds_va = JellyfishDataset(va_p, va_l, transform=get_transforms(False))
    ds_te = JellyfishDataset(te_p, te_l, transform=get_transforms(False))

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = JellyfishModel(num_classes=len(mapping))

    ckpt = ModelCheckpoint(
        dirpath='checkpoints',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename='jelly-{epoch:02d}-{val_loss:.2f}'
    )
    es = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus>0 else 'cpu',
        devices=args.gpus if args.gpus>0 else None,
        callbacks=[es, ckpt],
        enable_progress_bar=True,
    )
    trainer.fit(model, dl_tr, dl_va)
    trainer.test(model, dl_te)

    # save final
    torch.save(model.state_dict(), 'final_model.pth')

if __name__ == '__main__':
    main()

