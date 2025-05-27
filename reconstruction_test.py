import os
# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
torch.cuda.is_available = lambda: False

import argparse
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import matplotlib.pyplot as plt

from train_models import VAE, vae_loss, SpriteDataset, GANGenerator, Discriminator

# -------------------------------
# Configuration defaults
# -------------------------------
DEVICE        = torch.device('cpu')
DEFAULT_IMG_SIZE    = 64
DEFAULT_NUM_OVERFIT = 8
DEFAULT_VAE_EPOCHS  = 100
DEFAULT_GAN_EPOCHS  = 200
DEFAULT_BATCH_SIZE  = 4
DEFAULT_LR_VAE      = 1e-3
DEFAULT_LR_G        = 2e-4
DEFAULT_LR_D        = 1e-4
DEFAULT_REPORT_INT  = 50

# -------------------------------
# Utility: set seeds for reproducibility
# -------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------------------
# VAE Overfit Test
# -------------------------------
def overfit_vae(args):
    # seed
    set_seed(args.seed)

    # dataset subset
    full_ds = SpriteDataset(args.folder, img_size=args.img_size)
    indices = list(range(min(args.num_overfit, len(full_ds))))
    subset  = Subset(full_ds, indices)
    loader  = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

    # model & optimizer
    vae = VAE(latent_dim=args.latent_dim).to(DEVICE)
    opt = optim.Adam(vae.parameters(), lr=args.lr_v)

    # training
    for epoch in range(1, args.epochs_v + 1):
        total = 0.0
        for x in loader:
            x = x.to(DEVICE)
            recon, mu, lv = vae(x)
            loss = vae_loss(recon, x, mu, lv)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print(f"[VAE Epoch {epoch}/{args.epochs_v}] avg loss/img: {total/args.num_overfit:.4f}")

    # visualize originals vs reconstructions
    x_orig = next(iter(loader)).to(DEVICE)
    recon, _, _ = vae(x_orig)
    x_orig = x_orig.cpu(); recon = recon.detach().cpu()

    n = x_orig.size(0)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    for i in range(n):
        img_o = x_orig[i].permute(1,2,0).numpy() * 0.5 + 0.5
        img_r = recon[i].permute(1,2,0).numpy()   * 0.5 + 0.5
        axes[0,i].imshow(img_o); axes[0,i].axis('off')
        axes[1,i].imshow(img_r); axes[1,i].axis('off')
    axes[0,0].set_ylabel("Original", fontsize=12)
    axes[1,0].set_ylabel("Reconstruction", fontsize=12)
    plt.suptitle("VAE Overfit Reconstruction")
    plt.tight_layout()
    plt.show()

# -------------------------------
# GAN Overfit Test
# -------------------------------
def overfit_gan(args):
    # seed
    set_seed(args.seed)

    # dataset subset
    full_ds = SpriteDataset(args.folder, img_size=args.img_size)
    indices = list(range(min(args.num_overfit, len(full_ds))))
    subset  = Subset(full_ds, indices)
    loader  = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

    # models, loss, optimizers
    G = GANGenerator(latent_dim=args.latent_dim_g).to(DEVICE)
    D = Discriminator().to(DEVICE)
    criterion = torch.nn.BCELoss()
    optG = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5,0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5,0.999))

    # fixed labels
    for epoch in range(1, args.epochs_g + 1):
        for real in loader:
            real = real.to(DEVICE)
            bs   = real.size(0)
            valid = torch.full((bs,1), 0.9, device=DEVICE)
            fake  = torch.full((bs,1), 0.1, device=DEVICE)

            # D step
            z = torch.randn(bs, args.latent_dim_g, device=DEVICE)
            fake_imgs = G(z).detach()
            d_loss = 0.5*(criterion(D(real), valid) + criterion(D(fake_imgs), fake))
            optD.zero_grad(); d_loss.backward(); optD.step()

            # G step
            z2 = torch.randn(bs, args.latent_dim_g, device=DEVICE)
            g_loss = criterion(D(G(z2)), valid)
            optG.zero_grad(); g_loss.backward(); optG.step()

        if epoch == 1 or epoch % args.report_int == 0:
            print(f"[GAN Epoch {epoch}/{args.epochs_g}] D_loss:{d_loss.item():.4f} G_loss:{g_loss.item():.4f}")

    # visualize real vs fake
    originals = next(iter(loader))[:min(args.num_overfit, args.batch_size)].cpu()
    originals = originals * 0.5 + 0.5
    with torch.no_grad():
        z = torch.randn(originals.size(0), args.latent_dim_g, device=DEVICE)
        fakes = G(z).cpu()
    fakes = (fakes + 1) * 0.5

    n = originals.size(0)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    for i in range(n):
        axes[0,i].imshow(originals[i].permute(1,2,0).numpy()); axes[0,i].axis('off')
        axes[1,i].imshow(fakes[i].permute(1,2,0).numpy()); axes[1,i].axis('off')
    axes[0,0].set_ylabel("Real", fontsize=12)
    axes[1,0].set_ylabel("Fake", fontsize=12)
    plt.suptitle("GAN Overfit Test")
    plt.tight_layout()
    plt.show()

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Overfit reconstruction tests for VAE and GAN models.")
    parser.add_argument("--model", choices=["vae","gan"], required=True)
    parser.add_argument("--folder", default="sprite_dataset")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--num_overfit", type=int, default=DEFAULT_NUM_OVERFIT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)

    # VAE args
    parser.add_argument("--epochs_v", type=int, default=DEFAULT_VAE_EPOCHS)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--lr_v", type=float, default=DEFAULT_LR_VAE)

    # GAN args
    parser.add_argument("--epochs_g", type=int, default=DEFAULT_GAN_EPOCHS)
    parser.add_argument("--latent_dim_g", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=DEFAULT_LR_G)
    parser.add_argument("--lr_d", type=float, default=DEFAULT_LR_D)
    parser.add_argument("--report_int", type=int, default=DEFAULT_REPORT_INT)

    args = parser.parse_args()
    if args.model == "vae":
        overfit_vae(args)
    else:
        overfit_gan(args)

if __name__ == '__main__':
    main()
