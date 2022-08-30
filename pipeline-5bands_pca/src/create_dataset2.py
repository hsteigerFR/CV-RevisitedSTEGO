import os
import random
import time
import argparse

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
    help="Output path for PCA model",
    type=str,
    required=True)
parser.add_argument(
    "-set_p",
    "--dataset_percent",
    help="Percentage of the data that will be used",
    type=float,
    default=0.25)
parser.add_argument(
    "-img_p",
    "--image_percent",
    help="Percentage of each picture that will be processed ",
    type=float,
    default=0.25)
args = parser.parse_args()

if __name__ == "__main__":
    print("Creating output directory ...")
    output_folder = os.path.dirname(args.output_path)
    os.makedirs(output_folder, exist_ok=True)

    print("Importing multispectral images ...")
    L_all = [os.path.join(args.input_path, element)
             for element in os.listdir(args.input_path)]

    image_sample_size = int(args.dataset_percent * len(L_all))
    L = random.sample(L_all, image_sample_size)
    R = []
    for i in tqdm.tqdm(range(len(L))):
        A = np.load(L[i])
        A = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
        sample = np.random.choice(
            A.shape[0], int(
                args.image_percent * A.shape[0]))
        R.append(A[sample, :])
    all_points = np.concatenate(tuple(R))

    print("Calculating PCA ...")
    print(f"This should take less than : {len(all_points)/A.shape[0]} s")
    pca = PCA(n_components=3)
    pca.fit(all_points)

    print("Exporting Model ...")
    model_output_path = os.path.join(args.output_path)
    pickle.dump(pca, open(model_output_path, 'wb'))
