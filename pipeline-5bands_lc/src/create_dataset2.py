import argparse
import os
import random
import sys
import time

import cv2
import tqdm
import pickle
import numpy as np

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_path",
    help="Input path for aligned dataset",
    type=str,
    required=True)
parser.add_argument(
    "-o",
    "--output_path",
    help="Output path for processed dataset",
    type=str,
    required=True)
parser.add_argument(
    "-m",
    "--lc_matrix_path",
    help="Path of the matrix used for dimensionality reduction with linear combination",
    type=str,
    required=True)
parser.add_argument(
    "-output_s",
    "--output_size",
    help="Output size for pictures processed with LC. It should be about the same size as the resolution of STEGO.",
    type=int,
    default=300)
parser.add_argument(
    "-padding_s",
    "--padding_step",
    help="Padding step for slicing. It should be a divisor of STEGO resolution.",
    type=int,
    default=100)
args = parser.parse_args()

if __name__ == "__main__":
    print("Creating output directory ...")
    os.makedirs(args.output_path, exist_ok=True)

    print("Importing LC matrix ...")
    M = np.genfromtxt(args.lc_matrix_path, delimiter=",")
    if (np.linalg.norm(M, axis=1, ord=1) - np.array([1, 1, 1])).any():
        print("Error : Invalid LC matrix, the sum of the rows should equal one.")
        sys.exit(-1)

    print("Importing multispectral images and exporting processed dataset ...")
    win_s = args.output_size
    pad_s = args.padding_step

    L = [os.path.join(args.input_path, element)
         for element in os.listdir(args.input_path)]
    n = 0
    for i in tqdm.tqdm(range(len(L))):
        A = np.load(L[i])
        H, W, _ = A.shape
        for u in range(0, W, pad_s):
            for v in range(0, H, pad_s):
                loc_img_full = A[v:v + win_s, u:u + win_s, :]
                if loc_img_full.shape[:2] == (win_s, win_s):
                    loc_img_reduced = loc_img_full.dot(
                        np.transpose(M)).astype(np.uint8)
                    export_path = os.path.join(args.output_path, f"{n}.png")
                    cv2.imwrite(export_path, loc_img_reduced)
                    n += 1
