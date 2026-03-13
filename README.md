# deepfake-detection-resnet-gan

> Comparative study of supervised (ResNet-18 + CBAM) vs. unsupervised (GAN anomaly detection) paradigms for deepfake detection on 290,335 facial images — with a clear winner and a clear explanation of why.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange) ![Dataset](https://img.shields.io/badge/dataset-290%2C335%20images-lightgrey) ![Best ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9653-brightgreen) ![License](https://img.shields.io/badge/license-MIT-green)

---

## The Question

Two fundamentally different learning strategies exist for deepfake detection:

1. **Supervised**: train on both real and fake images, learn the discriminative boundary explicitly
2. **Anomaly detection**: train only on real images, flag fakes as deviations from learned normality

The second approach is theoretically appealing — it doesn't require labeled fake data and could generalize to unseen generation techniques. This project tests whether that theoretical appeal holds empirically.

**Short answer: it doesn't.** But understanding why is the contribution.

---

## Results

### Supervised Classification — 5 Variants (ResNet-18)

| Variant | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Baseline (from scratch) | 85.04% | 0.8358 | 0.9460 | 0.9440 |
| Transfer Learning (ImageNet) | 84.40% | **0.8645** | 0.9428 | 0.9114 |
| CBAM Attention (from scratch) | 86.16% | 0.8545 | 0.9434 | 0.9414 |
| CutMix Regularization | 86.26% | 0.8572 | 0.9392 | 0.9464 |
| **CBAM + Dropout** | **86.92%** | 0.8568 | **0.9616** | **0.9653** |

### Supervised vs. GAN Anomaly Detection

| Approach | Accuracy | F1 | ROC-AUC | EER |
|---|---|---|---|---|
| **CBAM + Dropout (Supervised)** | **86.92%** | **0.8568** | **0.9616** | **0.1325** |
| GAN Discriminator (Anomaly) | 47.20% | 0.4112 | 0.4701 | 0.5217 |

The GAN approach performs **below random chance** (ROC-AUC < 0.5). An EER of 0.52 means no usable decision threshold exists.

---

## Architecture

### Supervised Pipeline
```
Input (224×224 RGB)
    → ResNet-18 backbone
        → layer1 → layer2 → layer3 → layer4
                                         ↓
                                    CBAM Module
                                    ├── Channel Attention (MLP, ratio=16)
                                    └── Spatial Attention (7×7 Conv)
                                         ↓
                                  Global Average Pooling
                                         ↓
                                  Dropout (p=0.5)
                                         ↓
                                  Linear (512 → 2)
```

**Key design decisions:**
- CBAM placed after `layer4` to operate on high-level semantic features
- Dropout only in Variant 5 — its interaction with CBAM is what produces the best results (attention focuses features, dropout prevents over-reliance on them)
- Transfer learning used ImageNet normalization `mean=[0.485, 0.456, 0.406]`; scratch models used symmetric `[-1,1]` normalization — mixing these would invalidate the comparison
- StepLR scheduler: `gamma=0.1` every 7 epochs — learning rate reduction after epoch 7 produces a visible kink in the loss curve

**CutMix implementation:**
```python
def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0)).to(x.device)
    y_a, y_b = y, y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam
```

### GAN Anomaly Detection Pipeline

DCGAN architecture trained **exclusively on 70,001 authentic images**. After training, the discriminator's output logit is used as a "realness" score — high score = real, low score = fake.

Generator: latent vector (100-dim) → 5× transposed convolutions → 224×224 RGB (Tanh output)
Discriminator: 224×224 RGB → 6× convolutions (LeakyReLU, slope=0.2) → single logit

**Training dynamics (50 epochs):**
```
Epoch  1: Loss_D=0.7123, Loss_G=3.7792   # Balanced competition
Epoch 23: Loss_D=0.0010, Loss_G=10.3531  # Discriminator dominance
Epoch 46: Loss_D=0.0012, Loss_G=35.8950  # Generator collapse
Epoch 50: Loss_D=0.2770, Loss_G=9.8736   # Unstable final state
```

The discriminator learned to distinguish GAN-generated fakes from real images — but this is not the same as detecting deepfakes at test time.

---

## Dataset

[DeepFake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) — 290,335 balanced facial images

| Split | Real | Fake | Total |
|---|---|---|---|
| Train | 70,001 | 70,001 | 140,002 |
| Validation | 19,787 | 19,641 | 39,428 |
| Test | 55,524 | 55,381 | 110,905 |

---

## Interpretability — Grad-CAM

Grad-CAM visualization on the best model reveals that the network has learned semantically meaningful decision regions:

- **Real images**: attention distributed across natural facial features (eyes, nose, mouth contours)
- **Fake images**: concentrated activation on known artifact zones — iris boundaries, eyelid edges, face-background transitions, skin texture inconsistencies

Implementation selects `layer4[-1].conv2` as target, with automatic fallback when CBAM is present:
```python
layer4_children = list(model.layer4.children())
target_block = layer4_children[-1]
if target_block.__class__.__name__.lower() == "cbam":
    target_block = layer4_children[-2]
target_layer = target_block.conv2
```

---

## Tech Stack

- **Framework**: PyTorch + torchvision
- **Architecture**: ResNet-18, DCGAN
- **Attention**: CBAM (channel + spatial)
- **Augmentation**: CutMix, RandomCrop, ColorJitter
- **Interpretability**: Grad-CAM (torchcam)
- **Training**: Adam (lr=3e-4), CrossEntropyLoss, StepLR

---

## Quickstart
```bash
git clone https://github.com/francesco-univr/DeepfakeDetection-ComparativeAnalysis-SupervisedClassificationVsGAN-AnomalyDetection.git
cd DeepfakeDetection-ComparativeAnalysis-SupervisedClassificationVsGAN-AnomalyDetection
pip install -r requirements.txt
```
```bash
# Supervised training (best variant)
python -m mycode.train --root /path/to/dataset --epochs 10 --cbam --dropout

# GAN-based anomaly detection training
python train_gan.py --root /path/to/dataset --epochs 50

# Evaluation
python -m mycode.evaluate --model runs/model.pth --root /path/to/dataset

# Grad-CAM visualization
python -m mycode.cam_vis --model runs/model.pth --root /path/to/dataset
```

---

## Project Structure
```
├── mycode/
│   ├── train.py          # Supervised training loop (all 5 variants)
│   ├── evaluate.py       # Evaluation: accuracy, F1, ROC-AUC, PR-AUC
│   ├── cam_vis.py        # Grad-CAM visualization
│   ├── model.py          # ResNet-18 + CBAM architecture
│   └── dataset.py        # DataLoader with augmentation pipelines
├── train_gan.py          # DCGAN adversarial training
├── report/
│   └── Report.pdf        # Full technical report
└── requirements.txt
```

---

## Key Learnings

1. **GAN discriminators don't transfer to deepfake detection**: A discriminator trained to distinguish real faces from GAN-generated faces learns to detect GAN artifacts specifically — not the general concept of "synthetic image." At test time, deepfakes generated by different methods don't exhibit those same GAN artifacts, so the discriminator has no signal. The ROC-AUC below 0.5 confirms the scores are actually inverted relative to the task.

2. **Transfer learning can hurt precision-recall balance**: The ImageNet variant achieved the highest F1 (0.8645) but lower accuracy than the scratch baseline (84.40% vs 85.04%). ImageNet pre-training produces better-calibrated predictions at the 0.5 threshold but doesn't add discriminative power for this domain — the deepfake dataset is large enough (290k images) to learn from scratch effectively.

3. **Attention + regularization is complementary, not redundant**: CBAM alone improves accuracy to 86.16%. Dropout alone (not tested in isolation here) would reduce variance. Combined in Variant 5, CBAM focuses the model on artifact-rich facial regions while Dropout prevents over-reliance on those same regions — the interaction produces the best ROC-AUC jump (0.9434 → 0.9616), which is more meaningful than the accuracy gain for a detection task where threshold-independence matters.
