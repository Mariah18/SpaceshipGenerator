"""
Background Removal Using Rembg
Usage:
python remove_background.py --input augmented_dataset --output sprite_dataset
"""

import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import rembg

def process_directory(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process common image extensions
    img_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg'))
    if not img_files:
        print(f"No images found in '{input_dir}'")
        return

    for img_file in img_files:
        # Load image
        img = Image.open(img_file).convert('RGBA')
        # Convert to NumPy array
        input_array = np.array(img)
        # Remove background
        output_array = rembg.remove(input_array)
        # Convert back to PIL Image
        result_img = Image.fromarray(output_array)
        # Save as PNG to preserve transparency
        out_file = output_path / f"{img_file.stem}.png"
        result_img.save(out_file)

    print(f"Processed {len(img_files)} images to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove backgrounds using rembg")
    parser.add_argument("--input",  "-i", required=True, help="Input folder of images")
    parser.add_argument("--output", "-o", required=True, help="Output folder for BG-removed images")
    args = parser.parse_args()

    process_directory(args.input, args.output)
