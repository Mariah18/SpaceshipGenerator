import os
import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

def generate_augmentations(img: Image.Image):
    aug_list = []

    # Flips
    aug_list.append(('flip_h', img.transpose(Image.FLIP_LEFT_RIGHT)))
    aug_list.append(('flip_v', img.transpose(Image.FLIP_TOP_BOTTOM)))

    # Rotations
    for angle in (90, 180, 270):
        aug_list.append((f'rot{angle}', img.rotate(angle, expand=True).resize(img.size)))

    # Brightness jitter
    for factor in (0.8, 1.2):
        enhancer = ImageEnhance.Brightness(img)
        aug_list.append((f'bright{factor}', enhancer.enhance(factor)))

    # Contrast jitter
    for factor in (0.8, 1.2):
        enhancer = ImageEnhance.Contrast(img)
        aug_list.append((f'contrast{factor}', enhancer.enhance(factor)))

    # Translations (10% shift)
    w, h = img.size
    shift = int(0.1 * w)
    # Expand canvas by shift in each direction, then crop
    padded = ImageOps.expand(img, border=(shift, shift, shift, shift), fill=(0,0,0))
    for dx, dy in ((shift,0), (-shift,0), (0,shift), (0,-shift)):
        crop = padded.crop((shift+dx, shift+dy, shift+dx+w, shift+dy+h))
        aug_list.append((f'trans{dx}_{dy}', crop))

    # Saturation jitter
    for factor in (0.7, 1.3):
        aug_list.append((f'sat{factor}', ImageEnhance.Color(img).enhance(factor)))

    # Hue jitter via HSV channel shift
    hsv = img.convert('HSV')
    h, s, v = hsv.split()
    for shift in (-20, 20):
        h2 = h.point(lambda p: (p + shift) % 256)
        aug_list.append((f'hue{shift}', Image.merge('HSV', (h2, s, v)).convert('RGB')))

    # Gaussian blur
    for r in (1, 2):
        aug_list.append((f'blur{r}', img.filter(ImageFilter.GaussianBlur(radius=r))))

    # Posterize
    for bits in (2, 3):
        aug_list.append((f'posterize{bits}', ImageOps.posterize(img, bits)))

    # Solarize
    for thresh in (64, 128, 192):
        aug_list.append((f'solarize{thresh}', ImageOps.solarize(img, thresh)))

    # Additive Gaussian noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 8, arr.shape)  # σ=8 out of [0–255]
    noised = np.clip(arr + noise, 0, 255).astype(np.uint8)
    aug_list.append(('noise', Image.fromarray(noised)))

    # Shear in X and Y
    for shear in (-0.15, 0.15):  # 15% shear
        matrix = (1, shear, 0, shear, 1, 0)
        sheared = img.transform(img.size, Image.AFFINE, matrix, resample=Image.BILINEAR)
        aug_list.append((f'shear{int(shear*100)}', sheared))

    return aug_list

def main(input_dir, output_dir):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    png_files = list(input_path.glob('*.png'))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return

    for img_file in png_files:
        img = Image.open(img_file).convert('RGB')
        stem = img_file.stem

        # Save original
        img.save(output_path / f"{stem}_orig.png")

        # Save all augmentations
        for suffix, aug in generate_augmentations(img):
            aug_file = output_path / f"{stem}_{suffix}.png"
            aug.save(aug_file)

    total = len(list(output_path.glob('*.png')))
    print(f"Augmented {len(png_files)} originals to {total} images in '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment a sprite dataset.")
    parser.add_argument("--input",  type=str, default="sprite_dataset",
                        help="Input folder of PNG images")
    parser.add_argument("--output", type=str, default="augmented_dataset",
                        help="Output folder")
    args = parser.parse_args()
    main(args.input, args.output)
