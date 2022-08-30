import os
import argparse

import cv2
import tqdm
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_path",
    help="Input path for jpeg images folder",
    type=str,
    required=True)
parser.add_argument(
    "-o",
    "--output_path",
    help="Output path for training dataset",
    type=str,
    required=True)
parser.add_argument(
    "-output_s",
    "--output_size",
    help="Target output size for dataset pictures. It should be the same size as the resolution of STEGO.",
    type=int,
    default=300)
parser.add_argument(
    "-padding_s",
    "--padding_step",
    help="Padding step for ortho slicing. It should be a divisor of STEGO resolution.",
    type=int,
    default=100)
parser.add_argument("-n", "--normalize", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    print("Create output directory ...")
    os.makedirs(args.output_path, exist_ok=True)

    print("Getting all jpeg images ...")
    jpeg_images = []
    all_folders = [
        os.path.join(
            args.input_path,
            element) for element in os.listdir(
            args.input_path) if os.path.isdir(
                os.path.join(
                    args.input_path,
                    element))]
    for folder in all_folders:
        for element in os.listdir(folder):
            if os.path.splitext(element)[-1] == ".JPG":
                jpeg_images.append(os.path.join(folder, element))

    print("Slicing and exporting dataset ...")
    win_size = args.output_size
    for i in tqdm.tqdm(range(len(jpeg_images))):
        img = cv2.imread(jpeg_images[i])
        n = 1
        for u in range(0, img.shape[1], win_size):
            for v in range(0, img.shape[0], win_size):
                loc_img = img[v:v + win_size, u:u + win_size, :]
                if loc_img.shape[:2] == (win_size, win_size):
                    export_path = os.path.join(
                        args.output_path, f"{i}-{n:02d}.png")
                    n += 1
                    cv2.imwrite(export_path, loc_img)
