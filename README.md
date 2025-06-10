# JellyFishVision

A PyTorch-Lightning pipeline for jellyfish species classification with CBAM, custom augmentations (underwater/beach effects), TabPFN, SHAP, and DCA support.

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
