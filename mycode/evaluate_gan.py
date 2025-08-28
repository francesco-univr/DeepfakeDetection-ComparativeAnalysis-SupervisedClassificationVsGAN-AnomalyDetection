import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_fscore_support, average_precision_score, accuracy_score,
    precision_recall_curve
)

from mycode.dataset import DeepFakeDataset
from mycode.gan_models import Discriminator  # Import the Discriminator model

# GAN discriminator evaluation script
# This script evaluates a trained GAN Discriminator as a deepfake detector.

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the test dataset
    
    try:
        test_ds = DeepFakeDataset(
            root=args.root,
            split="test",
            use_imagenet_stats=False,  # GANs are usually trained with [-1, 1] normalization
            size=args.size
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    print(f"Found {len(test_ds)} samples in the test set.")

    # Load the trained discriminator
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return
    
    # Create an instance of our Discriminator as it was during training (1 output class)
    # and then load its saved weights.
    model = Discriminator(num_classes=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Run Inference
    # The process is identical to the supervised evaluation
    all_true = []
    all_prob = []

    print("Starting evaluation on the test set...")
    with torch.no_grad():
        for x, y in tqdm(test_dl, desc="Evaluating", unit="batch"):
            x = x.to(device)
            logits = model(x)
            # Discriminator outputs a single logit representing "realness".
            # Apply a sigmoid to get the probability of being "Real".
           
            probs = 1 - torch.sigmoid(logits).squeeze()

            all_true.append(y.cpu().numpy())
            all_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_score = np.concatenate(all_prob, axis=0)

    # Compute and display metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = float(fpr[idx_eer])
    thr_eer = float(thresholds[idx_eer]) if idx_eer < len(thresholds) else 0.5
    y_pred_05 = (y_score > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_05) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_05, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(y_true, y_score)

    print("\n--- GAN Discriminator Evaluation Results ---")
    print(f"Accuracy (thr=0.5): {acc:.2f}%")
    print(f"Precision/Recall/F1 (thr=0.5): {prec:.4f} / {rec:.4f} / {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"EER: {eer:.4f}  (thr≈{thr_eer:.3f})")
    print("------------------------------------------\n")

    # Generate and save result plots 
    outdir = args.outdir or os.path.dirname(args.model) or "."
    os.makedirs(outdir, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_05)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title("Confusion Matrix (thr=0.5)")
    cm_path = os.path.join(outdir, "confusion_matrix_gan.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.legend()
    roc_path = os.path.join(outdir, "roc_curve_gan.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to: {roc_path}")

    # PR curve
    P, R, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(R, P, label=f"AP = {pr_auc:.3f}")
    plt.title("Precision-Recall Curve")
    plt.legend()
    pr_path = os.path.join(outdir, "pr_curve_gan.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"PR curve saved to: {pr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained GAN Discriminator as a deepfake detector.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained Discriminator's .pth file.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--size", type=int, default=224, help="Input image size.")
    parser.add_argument("--outdir", type=str, default="",
                        help="Directory to save plots (default: model's directory).")
    args = parser.parse_args()
    main(args)