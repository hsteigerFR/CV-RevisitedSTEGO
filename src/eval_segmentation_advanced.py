import os
import shutil
import argparse
from itertools import chain
from math import sqrt

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool, freeze_support
import hydra
import seaborn as sns
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter

saved_data = defaultdict(list)
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    help="Path to input image file or input image folder",
    type=str,
    required="True")
parser.add_argument(
    "-m",
    "--model",
    help="Path to model .ckpt file",
    type=str,
    required="True")
args = parser.parse_args()


def pre_process():
    print("Getting model ...")
    model = LitUnsupervisedSegmenter.load_from_checkpoint(args.model)
    res = model.cfg["res"]
    model_name = model.cfg["experiment_name"]

    print("Generating dataset ...")
    loc_path = os.path.join(model_name, "imgs", "loc")
    try:
        os.makedirs(loc_path)
    except FileExistsError:
        shutil.rmtree(model_name)
        os.makedirs(loc_path)
    if os.path.isdir(args.input):
        L = [os.path.join(args.input, element)
             for element in os.listdir(args.input)]
        for image_path in L:
            image_name = os.path.basename(image_path)
            shutil.copy(image_path, os.path.join(loc_path, image_name))
    else:
        image_name = os.path.basename(args.input)
        shutil.copy(args.input, os.path.join(loc_path, image_name))

    print("Importing dataset ...")
    batch_size = 1
    num_workers = 1
    n_pictures = len(os.listdir(loc_path))
    if n_pictures > 1:
        print(f"{n_pictures} images detected.")
    else:
        print(f"{n_pictures} image detected.")

    test_dataset = ContrastiveSegDataset(
        pytorch_data_dir=".",
        dataset_name="directory",
        crop_type=None,
        image_set="loc",
        transform=get_transform(res, False, "center"),
        target_transform=get_transform(res, True, "center"),
        cfg=model.cfg,
    )

    test_loader = DataLoader(test_dataset, batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True, collate_fn=flexible_collate)
    batch_nums = torch.tensor([n // (batch_size)
                              for n in range(0, n_pictures)])
    batch_offsets = torch.tensor([n % (batch_size)
                                 for n in range(0, n_pictures)])

    return model, test_loader, batch_nums, batch_offsets


def evaluate(model, test_loader, batch_nums, batch_offsets):
    global saved_data
    model.eval().cuda()
    par_model = model.net
    print("Running evaluation ...")

    for i, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            img = batch["img"].cuda()
            label = batch["label"].cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(code,
                                 label.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False)
            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
            cluster_probs = model.cluster_probe(code, 2, log_probs=True)

            linear_preds = linear_probs.argmax(1)
            cluster_preds = cluster_probs.argmax(1)
            if i in batch_nums:
                matching_offsets = batch_offsets[torch.where(batch_nums == i)]
                for offset in matching_offsets:
                    saved_data["cluster_probs"].append(
                        cluster_probs.cpu()[offset].unsqueeze(0))
                    saved_data["cluster_preds"].append(
                        cluster_preds.cpu()[offset].unsqueeze(0))
                    saved_data["img"].append(img.cpu()[offset].unsqueeze(0))


def img_unpack(x):
    x = x.unsqueeze(0)
    x_array = np.array(unnorm(x).squeeze(0).cpu())
    I = np.zeros((x_array.shape[2], x_array.shape[2], 3))
    I[:, :, 0] = x_array[0, 0, :, :] * 255  # R
    I[:, :, 1] = x_array[0, 1, :, :] * 255  # G
    I[:, :, 2] = x_array[0, 2, :, :] * 255  # B
    return I.astype(np.uint8)


def make_colormap(cluster_preds, n):
    dict_color = {}
    for k in range(n):
        dict_color[k] = np.random.randint(0, 256, 3)

    I = np.zeros(cluster_preds.shape + (3,))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            index = cluster_preds[i, j]
            I[i, j] = dict_color[index]
    return I


def kullback_leibler(probs, i, j):
    n_classes = probs.shape[1]
    res = 0
    for k in range(n_classes):
        res += np.exp(probs[0, k, i, j]) * \
            np.log(np.exp(probs[0, k, i, j]) / (1 / n_classes))
    return res


def correlation_matrix(probs):
    win_size = probs.shape[2]
    n_classes = probs.shape[1]
    cor_mat = np.zeros((n_classes, n_classes))

    for k1 in range(n_classes):
        for k2 in range(n_classes):
            mu_k1 = np.mean(probs[0, k1, :, :])
            mu_k2 = np.mean(probs[0, k2, :, :])
            for i in range(win_size):
                for j in range(win_size):
                    cor_mat[k1, k2] += (probs[0, k1, i, j] -
                                        mu_k1) * (probs[0, k2, i, j] - mu_k2)
            cor_mat[k1, k2] /= (win_size * win_size)
    return cor_mat


def show(saved_data):
    for j in range(len(saved_data["cluster_probs"])):
        probs = np.array(saved_data["cluster_probs"][j])
        n_classes = probs.shape[1]
        step = int(np.sqrt(n_classes))
        x = n_classes // step + 1 * (n_classes % step > 0)
        y = step

        cluster_preds = np.array(saved_data["cluster_preds"][j])[0, :, :]
        cluster_preds = make_colormap(
            cluster_preds,
            n_classes).astype(
            np.uint8)
        img = img_unpack(saved_data["img"][j])

        # Classes
        f, ax = plt.subplots(x, y)
        for i in range(n_classes):
            if x == 1 or y == 1:
                ax[i // step].set_axis_off()
                ax[i // step].imshow(img)
                ax[i // step].imshow(np.exp(probs[0, i, :, :]), alpha=0.5,
                                     vmin=0, vmax=np.max(np.exp(probs[0, i, :, :])))
                ax[i // step].title.set_text(f"Class {i}")

            else:
                ax[i // step, i % step].set_axis_off()
                ax[i // step, i % step].imshow(img)
                ax[i // step,
                   i % step].imshow(np.exp(probs[0,
                                                 i,
                                                 :,
                                                 :]),
                                    alpha=0.5,
                                    vmin=0,
                                    vmax=np.max(np.exp(probs[0,
                                                             i,
                                                             :,
                                                             :])))
                ax[i // step, i % step].title.set_text(f"Class {i}")
        plt.show()
        plt.close(f)

        # Covariance
        cov_mat = correlation_matrix(probs)
        f = plt.figure()
        plt.imshow(cov_mat, vmin=-0.3, vmax=0.3, cmap="jet")
        plt.title("Covariance")
        plt.show()
        plt.close(f)

        # Clusters
        f = plt.figure()
        plt.imshow(img)
        plt.imshow(cluster_preds, alpha=0.8)
        plt.title("Clusters")
        plt.show()

        # Uncertainty
        kl_array = np.zeros((probs.shape[2], probs.shape[2]))
        for i in range(probs.shape[2]):
            for j in range(probs.shape[2]):
                kl_array[i, j] = kullback_leibler(probs, i, j)
        kl_array = 1 - (kl_array / np.max(kl_array))
        my_metric = np.sum(kl_array) / (probs.shape[2] * probs.shape[2])

        f = plt.figure()
        plt.imshow(img)
        plt.imshow(kl_array, alpha=0.55)
        plt.title(f"KL Uncertainty : {my_metric}")
        plt.show()
        plt.close(f)


if __name__ == "__main__":
    model, test_loader, batch_nums, batch_offsets = pre_process()
    evaluate(model, test_loader, batch_nums, batch_offsets)
    shutil.rmtree(model.cfg["experiment_name"])
    show(saved_data)
