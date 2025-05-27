import os
import argparse
import csv
import numpy as np
from PIL import Image
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import tensorflow as tf

def symmetry_score(img: np.ndarray) -> float:
    flipped = np.fliplr(img)
    mse = np.mean((img - flipped) ** 2)
    norm = np.mean(img**2) + 1e-8
    return max(0.0, 1.0 - mse / norm)

def sharpness_score(img_gray: np.ndarray) -> float:
    lap = laplace(img_gray)
    return float(np.mean(np.abs(lap)))

def ssim_score(img: np.ndarray) -> float:
    img_f    = img_as_float(img)
    flipped  = np.fliplr(img_f)
    vals = []
    for c in range(img_f.shape[2]):
        vals.append(ssim(img_f[...,c], flipped[...,c], data_range=1.0))
    return float(np.mean(vals))

def entropy_score(img_gray: np.ndarray) -> float:
    return float(shannon_entropy(img_gray))

def foreground_area_score(img_gray: np.ndarray, thresh: float = 0.1) -> float:
    fg = img_gray > thresh
    return float(np.mean(fg))

def evaluate_image(path: str) -> dict:
    im   = Image.open(path).convert('RGB')
    arr  = np.array(im).astype(np.float32) / 255.0
    gray = rgb2gray(arr)
    return {
        'filename': os.path.basename(path),
        'symmetry_mse':   symmetry_score(arr),
        'ssim_symmetry':  ssim_score(arr),
        'sharpness':      sharpness_score(gray),
        'entropy':        entropy_score(gray),
        'foreground_area':foreground_area_score(gray),
    }

def main(input_dir: str, output_csv: str):
    # find PNGs
    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith('.png')
    ])
    if not files:
        raise RuntimeError(f"No PNGs in {input_dir}")

    # CSV header
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'filename', 'symmetry_mse', 'ssim_symmetry',
            'sharpness', 'entropy', 'foreground_area'
        ]
        writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer_csv.writeheader()

        # evaluate & log
        for step, path in enumerate(files):
            scores = evaluate_image(path)
            writer_csv.writerow(scores)
            print(f"Eval {scores['filename']} → "
                  f"sym={scores['symmetry_mse']:.3f}, "
                  f"ssim={scores['ssim_symmetry']:.3f}, "
                  f"sharp={scores['sharpness']:.3f}, "
                  f"ent={scores['entropy']:.3f}, "
                  f"fg={scores['foreground_area']:.3f}")

    print(f"\nResults CSV saved to {output_csv}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Evaluate sprites and log metrics to CSV"
    )
    p.add_argument('--input_dir',  type=str, required=True,
                   help='Folder of PNG sprites')
    p.add_argument('--output_csv', type=str, default='evaluation_results.csv',
                   help='Where to write the CSV')
    args = p.parse_args()
    main(args.input_dir, args.output_csv)
