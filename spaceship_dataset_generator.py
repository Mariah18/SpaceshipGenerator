import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

SPRITE_SIZE = 64  # final sprite resolution
DEMO_SAMPLES = 8
PROMPT = (
    "spaceship in pixelsprite style.png"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_pcg(image: Image.Image,
              brightness_range=(0.9, 1.1),
              contrast_range=(0.9, 1.1),
              sharpen=True):
    # Simple post-processing for quick visual polish
    image = ImageEnhance.Brightness(image).enhance(random.uniform(*brightness_range))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(*contrast_range))
    if sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=1))
    return image

# -------------------------------
# Dataset Creation Function
# -------------------------------

def generate_sprite_dataset(num_samples=500, output_dir='dataset_generated'):
    os.makedirs(output_dir, exist_ok=True)
    # Load pipeline for dataset generation
    DIFFUSION_MODEL = "PublicPrompts/All-In-One-Pixel-Model"
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(
            DIFFUSION_MODEL,
            revision='fp16', torch_dtype=torch.float16
        ).to(DEVICE)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(DIFFUSION_MODEL).to(DEVICE)
    pipe.enable_attention_slicing()

    for i in range(num_samples):
        out = pipe(
            prompt=PROMPT,
        )
        img = out.images[0]
        sprite = img.resize((SPRITE_SIZE, SPRITE_SIZE), Image.NEAREST)
        sprite = apply_pcg(sprite)
        path = Path(output_dir) / f"ship_{i:03d}.png"
        sprite.save(path)
        if (i+1) % 10 == 0:
            print(f"Saved {i+1}/{num_samples} sprites to '{output_dir}'")
    print(f"Dataset generation complete: {num_samples} images in '{output_dir}'")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == '__main__':
    print("Creating dataset of 500 sprites")
    generate_sprite_dataset()
