import argparse
import os

import cv2
import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i_ortho",
    "--ortho_path",
    help="Path to the orthophoto bands folder",
    type=str,
    required=True)
parser.add_argument(
    "-o",
    "--output_path",
    help="Output path for sliced orthophoto parts",
    type=str,
    required=True)
parser.add_argument(
    "-output_s",
    "--output_size",
    help="Output size for ortho slices. It should be the same size as the resolution of STEGO.",
    type=int,
    default=400)
parser.add_argument(
    "-padding_s",
    "--padding_step",
    help="Padding step for ortho slicing. It should be a divisor of STEGO resolution.",
    type=int,
    default=400)
args = parser.parse_args()


if __name__ == "__main__":
    print("Creating output folder ...")
    os.makedirs(args.output_path, exist_ok=True)

    print("Importing RGB orthophoto band ...")
    L = [os.path.join(args.ortho_path, element) for element in os.listdir(
        args.ortho_path) if os.path.splitext(element)[-1] == ".tif"]
    RGB_path = ""
    for element in L:
        if "transparent_mosaic_group1" in element:
            RGB_path = element
    RGB = cv2.imread(RGB_path).astype(np.uint8)

    print("Resizing bands ...")
    s = 1
    H, W = int(s * RGB.shape[0]), int(s * RGB.shape[1])
    RGB = cv2.resize(RGB, (W, H))

    print("Exploring and slicing orthophoto ...")
    win_s = args.output_size
    pad_s = args.padding_step
    for u in tqdm.tqdm(range(0, W, pad_s)):
        for v in range(0, H, pad_s):
            loc_slice = RGB[v:v + win_s, u:u + win_s, :]
            if loc_slice.any() and (
                    loc_slice.shape[0],
                    loc_slice.shape[1]) == (
                    win_s,
                    win_s):
                slice_path = os.path.join(
                    args.output_path, f"{u:05d}_{v:05d}.png")
                cv2.imwrite(slice_path, loc_slice)
