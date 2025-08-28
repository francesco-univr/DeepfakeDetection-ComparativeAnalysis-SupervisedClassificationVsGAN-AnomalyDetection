import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from datetime import datetime

# Import our new GAN models
from mycode.gan_models import Generator, Discriminator, weights_init

# Main training script for the GAN 
# This script trains a Generator and a Discriminator in an adversarial process.

def main(args):
    # Create the output directory and set the random seed for reproducibility.
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Dataset of real Images
    
    real_images_path = os.path.join(args.root, "train/Real")
    if not os.path.isdir(real_images_path):
        raise RuntimeError(f"Directory with real images not found at: {real_images_path}")

    dataset = dset.ImageFolder(root=os.path.join(args.root, "train"),
                               transform=transforms.Compose([
                                   transforms.Resize(args.size),
                                   transforms.CenterCrop(args.size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    
    # Simple filtering to get only 'Real' images (class 0)
    real_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
    real_dataset = torch.utils.data.Subset(dataset, real_indices)

    dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch,
                                             shuffle=True, num_workers=args.workers)
    print(f"Found {len(real_dataset)} real images for GAN training.")

    # Create the Generator and Discriminator
    netG = Generator(latent_dim=args.latent_dim).to(device)
    netD = Discriminator().to(device)

    # Apply the custom weights initialization
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Use BCEWithLogitsLoss which is more numerically stable than a Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()

    # Create fixed noise vector to see the Generator's progress over time
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup two separate Adam optimizers for each network
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    run_name = f"{args.out}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_name, exist_ok=True)
    img_list = []

    # Start training loop
    print("Starting GAN Training Loop...")
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for i, data in enumerate(pbar):
            # Train Discriminator
            netD.zero_grad()
            
            # Train with all-real batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Train with all-fake batch
            noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1) # Use .detach() to avoid gradients flowing into G
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(real_label) 
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            pbar.set_postfix({
                "Loss_D": f"{errD.item():.4f}",
                "Loss_G": f"{errG.item():.4f}",
            })

        # After each epoch, save a sample of generated images
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"{run_name}/fake_samples_epoch_{epoch:03d}.png", normalize=True)

        # Save model checkpoints
        if epoch % args.save_every == 0:
            torch.save(netD.state_dict(), f"{run_name}/discriminator_epoch_{epoch}.pth")
            torch.save(netG.state_dict(), f"{run_name}/generator_epoch_{epoch}.pth")

    print(f"Training finished. Models saved in {run_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GAN to learn the distribution of real images.")
    parser.add_argument('--root', required=True, help='Path to the dataset root directory (containing train/val/test).')
    parser.add_argument('--workers', type=int, default=0, help='Number of dataloader workers.')
    parser.add_argument('--batch', type=int, default=128, help='Batch size during training.')
    parser.add_argument('--size', type=int, default=224, help='Image size to train on.')
    parser.add_argument('--latent_dim', type=int, default=100, help='Size of the latent z vector.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers.')
    parser.add_argument('--seed', type=int, default=999, help='Random seed for reproducibility.')
    parser.add_argument('--out', type=str, default='runs_gan', help='Output directory for models and samples.')
    parser.add_argument('--save_every', type=int, default=5, help='Save a checkpoint every N epochs.')
    args = parser.parse_args()
    main(args)