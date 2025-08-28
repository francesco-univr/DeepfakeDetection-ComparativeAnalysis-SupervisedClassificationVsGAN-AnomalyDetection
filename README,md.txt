# Deepfake Detection: A Comparative Analysis of Supervised Classification and GAN-based Anomaly Detection

This project presents a comparative study between supervised classification and GAN-based anomaly detection approaches for deepfake detection, analyzing their effectiveness on a large-scale facial image dataset.

## Abstract
Two distinct deep learning paradigms was investigated: supervised classification using ResNet-18 with attention mechanisms and unsupervised anomaly detection using adversarial training. Findings reveal significant performance differences between these approaches.

## Key Results
- **Supervised Classification (CBAM + Dropout)**: 86.16% accuracy, 0.9434 ROC-AUC
- **GAN-based Anomaly Detection**: 47.20% accuracy, 0.4701 ROC-AUC

## Dataset
[DeepFake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) - 290,335 facial images

## Architecture Highlights
- ResNet-18 with CBAM attention mechanism
- Advanced data augmentation (CutMix)
- DCGAN-based anomaly detection
- Grad-CAM interpretability analysis

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```bash
# Supervised training
python -m mycode.train --root /path/to/dataset --epochs 10 --cbam --cutmix

# GAN-based training
python train_gan.py --root /path/to/dataset --epochs 50

# Evaluation
python -m mycode.evaluate --model runs/model.pth --root /path/to/dataset