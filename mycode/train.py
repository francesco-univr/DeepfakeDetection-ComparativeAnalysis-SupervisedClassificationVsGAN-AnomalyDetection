import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mycode.dataset import DeepFakeDataset
from mycode.model import get_model
from mycode.augment import cutmix as cutmix_fn 

# Set the random seed for all relevant libraries
def seed_all(seed: int = 42):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    seed_all(args.seed)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the training and validation datasets, applying the correct transformations
    tr_ds = DeepFakeDataset(args.root, split="train",
                            use_imagenet_stats=args.pretrained, size=args.size)
    try:
        va_ds = DeepFakeDataset(args.root, split="val",
                                use_imagenet_stats=args.pretrained, size=args.size)
    except RuntimeError:
        va_ds = DeepFakeDataset(args.root, split="validation",
                                use_imagenet_stats=args.pretrained, size=args.size)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    # Model
    model = get_model(arch=args.arch, pretrained=args.pretrained,
                      use_cbam=args.cbam, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Logging
    run_name = f"{args.out}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=run_name)

    best_acc = 0.0
    best_path = os.path.join(run_name, "best.pth")
    global_step = 0
    
    # Start the training loop
    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        pbar = tqdm(tr_dl, desc=f"Epoch {ep}/{args.epochs}", unit="batch")

        for inputs, labels in pbar:
            global_step += 1
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device)

            if args.cutmix:
                inputs, la, lb, lam = cutmix_fn(inputs, labels, alpha=1.0)
                outputs = model(inputs)
                loss = lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)
                preds = outputs.argmax(1)
                correct = (preds == la).sum().item()  
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                preds = outputs.argmax(1)
                correct = (preds == labels).sum().item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += correct
            total += inputs.size(0)

        
            writer.add_scalar("train/loss", loss.item(), global_step)
            pbar.set_postfix({
                "loss": f"{running_loss/total:.4f}",
                "acc": f"{running_corrects/total:.4f}"
            })

        scheduler.step()

        # Validation
        val_acc, val_loss = evaluate(model, va_dl, criterion, device)
        writer.add_scalar("val/loss", val_loss, ep)
        writer.add_scalar("val/acc", val_acc, ep)

        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "cbam": args.cbam,
                "pretrained": args.pretrained,
                "size": args.size
            }, best_path)

    writer.close()
    print(f"Best val acc: {best_acc:.4f} | saved to: {best_path}")

# Compute the average accuracy and loss
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        preds = outputs.argmax(1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += (preds == labels).sum().item()
        total += inputs.size(0)
    return total_correct / total, total_loss / total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Root Dataset")
    p.add_argument("--arch", type=str, default="resnet18", help="Backbone torchvision")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pretrained", action="store_true",
                   help="Use ImageNet pre-trained weights and normalization")
    p.add_argument("--cbam", action="store_true", help="Add CBAM to layer4")
    p.add_argument("--cutmix", action="store_true", help="CutMix in training")
    p.add_argument("--workers", type=int, default=0, help="Num. worker DataLoader")
    p.add_argument("--size", type=int, default=224, help="Input image size (square crop)")
    p.add_argument("--out", type=str, default="runs/experiment_1", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = p.parse_args()
    main(args)
