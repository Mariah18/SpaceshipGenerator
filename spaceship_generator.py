import os
import torch
from pathlib import Path
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import importlib.util
MODULE_PATH = os.path.join(os.getcwd(), 'train_models.py')
spec = importlib.util.spec_from_file_location('train_models', MODULE_PATH)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

# Extract model classes and configuration constants
VAE = train_mod.VAE
GANGenerator = train_mod.GANGenerator
LATENT_DIM_VAE = getattr(train_mod, 'LATENT_DIM_VAE', 128)
LATENT_DIM_GAN = getattr(train_mod, 'LATENT_DIM_GAN', 100)
OUTPUT_DIR = Path('checkpoints')  # train_models saves to 'checkpoints'

# Device
DEVICE = torch.device('cpu')

# Checkpoint paths
VAE_CKPT = OUTPUT_DIR / 'vae_spaceship.pth'
GAN_CKPT = OUTPUT_DIR / 'gan_spaceship.pth'

# Output dirs for generated
BASE_OUT_DIR = 'generated'
VAE_OUT_DIR = os.path.join(BASE_OUT_DIR, 'vae')
GAN_OUT_DIR = os.path.join(BASE_OUT_DIR, 'gan')

os.makedirs(VAE_OUT_DIR, exist_ok=True)
os.makedirs(GAN_OUT_DIR, exist_ok=True)

# Helper: display a grid of sprites
def show_batch(imgs, title=None):
    n = imgs.size(0)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    for i, img in enumerate(imgs):
        np_img = img.permute(1,2,0).cpu().numpy()
        axes[i].imshow(np_img)
        axes[i].axis('off')
    if title:
        fig.suptitle(title)
    plt.show()

# Helper: save batch of images
to_pil = ToPILImage()
def save_batch(imgs, out_dir, prefix):
    for idx, img in enumerate(imgs):
        pil = to_pil(img.cpu())
        path = os.path.join(out_dir, f"{prefix}_{idx:03d}.png")
        pil.save(path)

# Model loader for train_models
class SpaceshipGenerator:
    def __init__(self):
        # Load VAE
        self.vae = VAE().to(DEVICE)
        self.vae.load_state_dict(torch.load(VAE_CKPT, map_location=DEVICE))
        self.vae.eval()
        # Load GAN
        self.gan = GANGenerator().to(DEVICE)
        self.gan.load_state_dict(torch.load(GAN_CKPT, map_location=DEVICE))
        self.gan.eval()

    def sample_vae(self, n=4):
        z = torch.randn(n, LATENT_DIM_VAE, device=DEVICE)
        with torch.no_grad():
            out = self.vae.decode(z)
        return (out + 1) * 0.5  # from [-1,1] to [0,1]

    def sample_gan(self, n=4):
        z = torch.randn(n, LATENT_DIM_GAN, device=DEVICE)
        with torch.no_grad():
            out = self.gan(z)
        return (out + 1) * 0.5

if __name__ == '__main__':
    gen = SpaceshipGenerator()

    # Generate and save VAE sprites
    vae_imgs = gen.sample_vae(4)
    save_batch(vae_imgs, VAE_OUT_DIR, 'vae')
    show_batch(vae_imgs, title='VAE-generated Sprites (train_models)')
    print(f"Saved {len(vae_imgs)} VAE sprites to '{VAE_OUT_DIR}'")

    # Generate and save GAN sprites
    gan_imgs = gen.sample_gan(4)
    save_batch(gan_imgs, GAN_OUT_DIR, 'gan')
    show_batch(gan_imgs, title='GAN-generated Sprites (train_models)')
    print(f"Saved {len(gan_imgs)} GAN sprites to '{GAN_OUT_DIR}'")
