import os
import argparse

import cv2
import tqdm
import numpy as np
from multiprocessing import Pool, freeze_support
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_path",
    help="Input path to raw multispectral dataset",
    type=str,
    required=True)
parser.add_argument(
    "-o",
    "--output_path",
    help="Output path for aligned .npy multispectral images",
    type=str,
    required=True)
parser.add_argument(
    "-crop_s",
    "--crop_size",
    help="Amount of pixels to remove from the borders of aligned pictures",
    type=int,
    default=20)
parser.add_argument(
    "-pic_s",
    "--picture_size",
    help="Targeted size for aligned pictures (W,H). It should be divisble by the STEGO resolution used.",
    type=int,
    nargs=2,
    default=[
        1600,
         1200])
parser.add_argument(
    "-num_p",
    "--num_processes",
    help="Number of multiprocessing workers",
    type=int,
    default=15)
args = parser.parse_args()

crop_size = args.crop_size
output_size = (args.picture_size[0], args.picture_size[1])
num_processes = args.num_processes


def tif_image_preprocessing(tif_image_path):
    """This function undistorts and centers a spectral image."""
    str_mtd = os.popen(
        f"gdalinfo -mdd xml:XMP {tif_image_path} | findstr \"CalibratedOpticalCenterX CalibratedOpticalCenterY RelativeOpticalCenterX RelativeOpticalCenterY DewarpData VignettingData\"").readlines()
    centerX = float(str_mtd[0].split("\"")[1])
    centerY = float(str_mtd[1].split("\"")[1])
    # Relative optical center X
    decalX = round(float(str_mtd[2].split("\"")[1]))
    # Relative optical center Y
    decalY = round(float(str_mtd[3].split("\"")[1]))
    str_dewarp = str_mtd[4].split("\"")[1].split(";")[1]
    dewarpdata = [float(i) for i in str_dewarp.split(',')]
    vignettingData = [float(i) for i in str_mtd[5].split("\"")[1].split(',')]

    (fx, fy, cx, cy, k1, k2, p1, p2, k3) = dewarpdata
    # Camera Matrix
    mtx = np.array([[fx, 0, centerX + cx], [0, fy, centerY + cy], [0, 0, 1]])
    coeffs_dist = [k1, k2, p1, p2, k3]  # Distorsion coeffs
    dist = np.array(coeffs_dist)

    loc_im = cv2.imread(tif_image_path)[:, :, 0]
    undist = cv2.undistort(loc_im, mtx, dist)
    ty, tx = loc_im.shape
    undist_undecal = np.zeros(loc_im.shape)
    for x in range(tx):
        for y in range(ty):
            x_undecal = x + decalX
            y_undecal = y + decalY
            if (x_undecal in range(tx) and y_undecal in range(ty)):
                undist_undecal[y, x] = undist[y_undecal, x_undecal]
    corr_im = undist_undecal.astype(np.uint8)
    return corr_im


def process(element):
    """This function aligns a given set of spectral images into a multispectral image, and exports it"""
    i, jpeg_path = element
    B_path = jpeg_path.replace("0.JPG", "1.TIF")
    G_path = jpeg_path.replace("0.JPG", "2.TIF")
    R_path = jpeg_path.replace("0.JPG", "3.TIF")
    RE_path = jpeg_path.replace("0.JPG", "4.TIF")
    NIR_path = jpeg_path.replace("0.JPG", "5.TIF")

    B = cv2.resize(
        tif_image_preprocessing(B_path)[
            crop_size:-crop_size,
            crop_size:-crop_size],
        output_size)
    G = cv2.resize(
        tif_image_preprocessing(G_path)[
            crop_size:-crop_size,
            crop_size:-crop_size],
        output_size)
    R = cv2.resize(
        tif_image_preprocessing(R_path)[
            crop_size:-crop_size,
            crop_size:-crop_size],
        output_size)
    RE = cv2.resize(
        tif_image_preprocessing(RE_path)[
            crop_size:-crop_size,
            crop_size:-crop_size],
        output_size)
    NIR = cv2.resize(
        tif_image_preprocessing(NIR_path)[
            crop_size:-crop_size,
            crop_size:-crop_size],
        output_size)

    res = np.zeros((output_size[1], output_size[0], 5)).astype(np.uint8)
    res[:, :, 0] = B
    res[:, :, 1] = G
    res[:, :, 2] = R
    res[:, :, 3] = RE
    res[:, :, 4] = NIR
    export_path = os.path.join(args.output_path, f"{i}.npy")
    np.save(export_path, res)


def get_images():
    """This function retrieves the paths of all jpeg images in the raw image dataset."""
    jpeg_images = []
    all_folders = [
        os.path.join(
            args.input_path,
            element) for element in os.listdir(
            args.input_path) if os.path.isdir(
                os.path.join(
                    args.input_path,
                    element))]
    n = 0
    for folder in all_folders:
        for element in os.listdir(folder):
            if os.path.splitext(element)[-1] == ".JPG":
                jpeg_images.append((n, os.path.join(folder, element)))
                n += 1
    return jpeg_images


if __name__ == "__main__":
    print("Creating output folder ...")
    os.makedirs(args.output_path, exist_ok=True)

    print("Getting jpeg image list ...")
    jpeg_images = get_images()

    print("Assembling and exporting multispectral images ...")
    with Pool(num_processes) as p:
        r = list(
            tqdm.tqdm(
                p.imap(
                    process,
                    jpeg_images),
                total=len(jpeg_images)))
