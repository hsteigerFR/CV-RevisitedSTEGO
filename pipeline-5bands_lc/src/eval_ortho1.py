import argparse
import os

import cv2
import numpy as np
import pickle
import tqdm

from matplotlib import pyplot as plt

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
    help="Output path for dataset processed with PCA",
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
    help="Output size for ortho slices. It should be the same size as the resolution of STEGO.",
    type=int,
    default=100)
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

    print("Importing LC matrix ...")
    M = np.genfromtxt(args.lc_matrix_path, delimiter=",")
    if (np.linalg.norm(M, axis=1, ord=1) - np.array([1, 1, 1])).any():
        print("Error : Invalid LC matrix, the sum of the rows should equal one.")
        sys.exit(-1)

    print("Importing orthophoto bands ...")
    L = [os.path.join(args.ortho_path, element) for element in os.listdir(
        args.ortho_path) if os.path.splitext(element)[-1] == ".tif"]
    B_path = ""
    for element in L:
        if "transparent_mosaic_blue.tif" in element:
            B_path = element
    G_path = B_path.replace("blue", "green")
    R_path = B_path.replace("blue", "red")
    RE_path = B_path.replace("blue", "red edge")
    NIR_path = B_path.replace("blue", "nir")

    B = cv2.imread(B_path).astype(np.uint8)[:, :, 0]
    G = cv2.imread(G_path).astype(np.uint8)[:, :, 0]
    R = cv2.imread(R_path).astype(np.uint8)[:, :, 0]
    RE = cv2.imread(RE_path).astype(np.uint8)[:, :, 0]
    NIR = cv2.imread(NIR_path).astype(np.uint8)[:, :, 0]

    print("Resizing bands ...")
    s = 1
    H, W = int(s * B.shape[0]), int(s * B.shape[1])
    B = cv2.resize(B, (W, H))
    G = cv2.resize(G, (W, H))
    R = cv2.resize(R, (W, H))
    RE = cv2.resize(RE, (W, H))
    NIR = cv2.resize(NIR, (W, H))

    print("Constructing multispectral array ...")
    multispec_ortho = np.zeros((B.shape + (5,)))
    multispec_ortho[:, :, 0] = B
    multispec_ortho[:, :, 1] = G
    multispec_ortho[:, :, 2] = R
    multispec_ortho[:, :, 3] = RE
    multispec_ortho[:, :, 4] = NIR
    B = G = R = RE = NIR = 0

    print("Exploring array and applying LC dimensionality reduction ...")
    win_s = args.output_size
    pad_s = args.padding_step
    for u in tqdm.tqdm(range(0, W, pad_s)):
        for v in range(0, H, pad_s):
            loc_slice = multispec_ortho[v:v + win_s, u:u + win_s, :]
            if loc_slice.any() and (
                    loc_slice.shape[0],
                    loc_slice.shape[1]) == (
                    win_s,
                    win_s):
                slice_path = os.path.join(
                    args.output_path, f"{u:05d}_{v:05d}.png")
                img_new = loc_slice.dot(np.transpose(M)).astype(np.uint8)
                cv2.imwrite(slice_path, img_new)
