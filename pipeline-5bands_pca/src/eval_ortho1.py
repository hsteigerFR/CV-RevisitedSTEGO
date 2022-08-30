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
    "-i_pca",
    "--PCA_path",
    help="Input path for PCA model",
    type=str,
    required=True)
parser.add_argument(
    "-i_norm",
    "--norm_path",
    help="Input path for normalization coefficients related to the PCA model",
    type=str,
    required=True)
parser.add_argument(
    "-o",
    "--output_path",
    help="Output path for dataset processed with PCA",
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
parser.add_argument("-n", "--normalize", action="store_true")
args = parser.parse_args()


def normalize_picture(img, mins, maxs):
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mins[i]) / (maxs[i] - mins[i])
    return (255 * img).astype(np.uint8)


def show_slice(loc_slice):
    f, ax = plt.subplots(2, 3)
    for i in range(5):
        ax[i // 3, i % 3].imshow(loc_slice[:, :, i])
    plt.show()


def apply_model(loc_slice, model, mins, maxs):
    img_reshape = np.reshape(
        loc_slice,
        (loc_slice.shape[0] *
         loc_slice.shape[1],
         loc_slice.shape[2]))
    img_reshape = model.transform(img_reshape)
    img_new = np.reshape(
        img_reshape,
        (loc_slice.shape[0],
         loc_slice.shape[1],
         3))
    if args.normalize:
        img_new = cv2.cvtColor(
            normalize_picture(
                img_new,
                mins,
                maxs).astype(
                np.uint8),
            cv2.COLOR_BGR2RGB)
    else:
        for i in range(0, 3):
            img_new[:,
                    :,
                    i] = 255 * (img_new[:,
                                        :,
                                        i] - np.min(img_new[:,
                                                            :,
                                                            i])) / (np.max(img_new[:,
                                                                                   :,
                                                                                   i]) - np.min(img_new[:,
                                                                                                        :,
                                                                                                        i]))
        img_new = cv2.cvtColor((img_new).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img_new


if __name__ == "__main__":
    print("Creating output folder ...")
    os.makedirs(args.output_path, exist_ok=True)

    print("Importing PCA model ...")
    model_path = args.PCA_path
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    norm_params_path = args.norm_path
    with open(norm_params_path, "rb") as f:
        my_dict = pickle.load(f)
    mins = my_dict["mins"]
    maxs = my_dict["maxs"]

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

    print("Exploring array and applying PCA...")
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
                img_new = apply_model(loc_slice, model, mins, maxs)
                cv2.imwrite(slice_path, img_new)
