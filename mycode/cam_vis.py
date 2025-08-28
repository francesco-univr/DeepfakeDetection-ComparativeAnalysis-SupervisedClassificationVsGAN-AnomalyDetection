import argparse
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from mycode.dataset import DeepFakeDataset
from mycode.model import get_model
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

# Load a model checkpoint, supporting both new and old formats for retrocompatibility
def load_checkpoint(model_path, device, fallback_arch="resnet18", fallback_cbam=False):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        arch = ckpt.get("arch", fallback_arch)
        use_cbam = ckpt.get("cbam", fallback_cbam)
        was_pretrained = ckpt.get("pretrained", False)
    else:
        # fallback for old checkpoints
        state_dict = ckpt
        arch = fallback_arch
        use_cbam = fallback_cbam
        was_pretrained = False
    return state_dict, arch, use_cbam, was_pretrained

# Reverse image normalization to allow for correct visualization
def denorm(t, use_imagenet_stats: bool):
    "Undo normalization for correct image visualization"
    if use_imagenet_stats:
        mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.5, 0.5, 0.5], device=t.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.5, 0.5, 0.5], device=t.device).view(1, 3, 1, 1)
    x = t * std + mean
    return x.clamp(0, 1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model Checkpoint
    force_cbam = "cbam" in os.path.basename(args.model).lower()
    state_dict, arch, use_cbam, was_pretrained = load_checkpoint(
        args.model, device, fallback_arch="resnet18", fallback_cbam=force_cbam
    )

    # Model
    model = get_model(use_cbam=use_cbam, arch=arch, pretrained=False).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Attention load_state_dict:")
        if missing: print("  missing:", missing)
        if unexpected: print("  unexpected:", unexpected)
    model.eval()

    # Set up Grad-CAM extractor
    # Automatically select the final convolutional layer of the model as the target
    layer4_children = list(model.layer4.children())
    target_block = layer4_children[-1]
    if target_block.__class__.__name__.lower() == "cbam":
        target_block = layer4_children[-2]
    if not hasattr(target_block, "conv2"):
        raise RuntimeError("Can’t find ‘conv2’ in the target block for Grad-CAM")
    target_layer = target_block.conv2
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # Dataset test
    use_imagenet_stats = args.pretrained or was_pretrained
    test_ds = DeepFakeDataset(
        root=args.root, split="test", use_imagenet_stats=use_imagenet_stats, size=args.size
    )
    # Take a random batch of N images
    bs = min(args.num_images, len(test_ds))
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=True, num_workers=0)
    images, labels = next(iter(test_dl))
    images = images.to(device)

    print(f"Display {bs} images from the test set")

    # Denormalize for proper visualization
    images_viz = denorm(images.clone(), use_imagenet_stats)
    pil_images = [transforms.ToPILImage()(img.cpu()) for img in images_viz]

    # CAM & predictions
    activation_maps, pred_classes = [], []
    with torch.no_grad():
        for i in range(bs):
            x = images[i].unsqueeze(0)
            out = model(x)
            pred = int(out.argmax(1).item())
            pred_classes.append(pred)

            # Grad-CAM for predicted class
            cam = cam_extractor(class_idx=pred, scores=out)[0].detach().cpu()  
            # normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
            activation_maps.append(cam)

    # Grid Visualization
    cols = 4
    rows = (bs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.reshape(rows, cols) if rows > 1 else axes.reshape(1, -1)

    for idx in range(bs):
        row, col = divmod(idx, cols)
        base = pil_images[idx]
        mask_pil = transforms.ToPILImage()(activation_maps[idx])
        result = overlay_mask(base, mask_pil, alpha=0.5)
        ax = axes[row, col]
        ax.imshow(result)
        ax.axis("off")
        true_label = "Fake" if int(labels[idx]) == 1 else "Real"
        pred_label = "Fake" if pred_classes[idx] == 1 else "Real"
        ax.set_title(f"True: {true_label} | Pred: {pred_label}",
                     color=("green" if true_label == pred_label else "red"))

    # Hide extra axes
    for idx in range(bs, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    plt.tight_layout()

    # Save work
    out_dir = os.path.dirname(args.model) or "."
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "grad_cam_visualization.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nDisplay Grad-CAM saved in: {plot_path}")

    cam_extractor.remove_hooks()

# Script entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="Path to the .pth file of the trained model")
    ap.add_argument("--root", type=str, required=True,
                    help="Root dataset directory")
    ap.add_argument("--num_images", type=int, default=8,
                    help="Number of images to display")
    ap.add_argument("--size", type=int, default=224,
                    help="Input side (crop)")
    ap.add_argument("--pretrained", action="store_true",
                    help="Use ImageNet normalization")
    args = ap.parse_args()
    main(args)

