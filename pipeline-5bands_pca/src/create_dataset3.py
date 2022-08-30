import os
import argparse

import cv2
import tqdm
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i_imgs",
    "--images_path",
    help="Input path for aligned dataset",
    type=str,
    required=True)
parser.add_argument(
    "-i_pca",
    "--PCA_path",
    help="Input path for PCA model",
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
    help="Output size for pictures processed with PCA. It should be the same size as the resolution of STEGO.",
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


def normalize_picture(img, mins, maxs):
    for i in range(3):
        img[:, :, i] = (img[:, :, i] - mins[i]) / (maxs[i] - mins[i])
    return (255 * img).astype(np.uint8)


if __name__ == "__main__":
    print("Create output directory ...")
    os.makedirs(args.output_path, exist_ok=True)

    print("Importing model ...")
    model_path = args.PCA_path
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model explicability per band : {model.explained_variance_ratio_}")

    print("Getting multispectral images ...")
    L_all = [
        os.path.join(
            args.images_path,
            element) for element in os.listdir(
            os.path.join(
                args.images_path))]

    if args.normalize:
        print("Getting normalization parameters ...")
        L_save = []
        min_L_C1 = []
        max_L_C1 = []

        min_L_C2 = []
        max_L_C2 = []

        min_L_C3 = []
        max_L_C3 = []
        for i in tqdm.tqdm(range(len(L_all))):
            img = np.load(L_all[i])
            img_reshape = np.reshape(
                img, (img.shape[0] * img.shape[1], img.shape[2]))
            img_reshape = model.transform(img_reshape)
            img_new = np.reshape(img_reshape, (img.shape[0], img.shape[1], 3))
            L_save.append(img_new)

            min_L_C1.append(np.min(img_new[:, :, 0]))
            max_L_C1.append(np.max(img_new[:, :, 0]))

            min_L_C2.append(np.min(img_new[:, :, 1]))
            max_L_C2.append(np.max(img_new[:, :, 1]))

            min_L_C3.append(np.min(img_new[:, :, 2]))
            max_L_C3.append(np.max(img_new[:, :, 2]))
        mins = (min(min_L_C1), min(min_L_C2), min(min_L_C3))
        maxs = (max(max_L_C1), max(max_L_C2), max(max_L_C3))

        print("Saving normalization parameters ...")
        my_dict = {"mins": mins, "maxs": maxs}
        with open(os.path.join(os.path.dirname(args.PCA_path), 'norm_params.sav'), 'wb') as f:
            pickle.dump(my_dict, f)

    print("Applying PCA model and exporting images ...")
    for i in tqdm.tqdm(range(len(L_all))):

        img = np.load(L_all[i])
        img_reshape = np.reshape(
            img, (img.shape[0] * img.shape[1], img.shape[2]))
        img_reshape = model.transform(img_reshape)
        img_new = np.reshape(img_reshape, (img.shape[0], img.shape[1], 3))

        if args.normalize:
            img_new = L_save[i]
            img_new = cv2.cvtColor(
                normalize_picture(
                    img_new,
                    mins,
                    maxs),
                cv2.COLOR_BGR2RGB)
        else:
            for i in range(0, 3):
                img_new[:, :, i] = (255 *
                                    (img_new[:, :, i] -
                                     np.min(img_new[:, :, i])) /
                                    (np.max(img_new[:, :, i]) -
                                        np.min(img_new[:, :, i])))
            img_new = cv2.cvtColor(img_new.astype(np.uint8), cv2.COLOR_BGR2RGB)

        win_size = args.output_size
        n = 1
        for u in range(0, img_new.shape[1], win_size):
            for v in range(0, img_new.shape[0], win_size):
                loc_img = img_new[v:v + win_size, u:u + win_size, :]
                if loc_img.shape[:2] == (win_size, win_size):
                    export_path = os.path.join(
                        args.output_path, f"{i}-{n:02d}.png")
                    n += 1
                    cv2.imwrite(export_path, loc_img)
