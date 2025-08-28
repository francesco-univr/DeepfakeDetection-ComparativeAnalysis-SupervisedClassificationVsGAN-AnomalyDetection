import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_fscore_support, average_precision_score, accuracy_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from mycode.dataset import DeepFakeDataset
from mycode.model import get_model


def load_checkpoint(model_path, device, fallback_arch="resnet18", fallback_cbam=False):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        arch = ckpt.get("arch", fallback_arch)
        use_cbam = ckpt.get("cbam", fallback_cbam)
    else:
        state_dict = ckpt
        arch = fallback_arch
        use_cbam = fallback_cbam
    return state_dict, arch, use_cbam


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset test 
    try:
        test_ds = DeepFakeDataset(
            root=args.root,
            split="test",
            use_imagenet_stats=args.pretrained,
            size=args.size
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        print("The test set was not found")
        return

    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    print(f"Find {len(test_ds)} samples on test set")

    # Model
    if not os.path.exists(args.model):
        print(f"Error: File mode '{args.model}' not found")
        return

    # Auto-detect
    force_cbam = args.cbam or ("cbam" in os.path.basename(args.model).lower())
    state_dict, arch_from_ckpt, cbam_from_ckpt = load_checkpoint(args.model, device,
                                                                 fallback_arch=args.arch,
                                                                 fallback_cbam=force_cbam)
    arch = arch_from_ckpt
    use_cbam = cbam_from_ckpt

    model = get_model(use_cbam=use_cbam, arch=arch, pretrained=False) 
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Unexpected keys in load_state_dict:")
        if missing:
            print("  Missing:", missing)
        if unexpected:
            print("  Unexpected:", unexpected)
    model.to(device).eval()

    # Inference
    all_true = []
    all_prob = []

    print("Evaluation test set")
    with torch.no_grad():
        for x, y in tqdm(test_dl, desc="Evaluating", unit="batch"):
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:, 1]  
            all_true.append(y.cpu().numpy())
            all_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_score = np.concatenate(all_prob, axis=0)

    # Metrics
    # ROC / AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # EER (Equal Error Rate)
    fnr = 1 - tpr
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    eer = float(fpr[idx_eer])
    thr_eer = float(thresholds[idx_eer]) if idx_eer < len(thresholds) else 0.5

    # Acc / PR-AUC / Precision-Recall-F1 thr 0.5
    y_pred_05 = (y_score > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_05) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_05, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(y_true, y_score)

    print("\n Evaluation sesults")
    print(f"Accuracy (thr=0.5): {acc:.2f}%")
    print(f"Precision/Recall/F1 (thr=0.5): {prec:.4f} / {rec:.4f} / {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"EER: {eer:.4f}  (thr≈{thr_eer:.3f})")
    print("-----------------------------\n")

    # Save plots
    outdir = args.outdir or os.path.dirname(args.model) or "."
    os.makedirs(outdir, exist_ok=True)

    # Confusion Matrix thr 0.5
    cm = confusion_matrix(y_true, y_pred_05)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (thr=0.5)")
    cm_path = os.path.join(outdir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion Matrix in: {cm_path}")

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    roc_path = os.path.join(outdir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC saved in{roc_path}")

    # PR curve
    from sklearn.metrics import precision_recall_curve
    P, R, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(R, P, lw=2, label=f"AP = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    pr_path = os.path.join(outdir, "pr_curve.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"PR saved in {pr_path}")

# Define and parse command-line arguments for running the evaluation
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="Path to the trained model's .pth file")
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory of the dataset")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for evaluation")
    ap.add_argument("--workers", type=int, default=0, help="Num. worker DataLoader.")
    ap.add_argument("--size", type=int, default=224, help="Input image size (square crop)")
    ap.add_argument("--pretrained", action="store_true",
                    help="ImageNet normalization stats")
    ap.add_argument("--arch", type=str, default="resnet18",
                    help="Fallback architecture")
    ap.add_argument("--cbam", action="store_true",
                    help="Force using CBAM")
    ap.add_argument("--outdir", type=str, default="",
                    help="Directory to save plots")
    args = ap.parse_args()
    main(args)

