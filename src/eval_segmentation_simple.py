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
parser.add_argument(
    "-cov",
    "--covariance",
    help="Enable covariance display",
    action='store_true')
parser.add_argument(
    "-l",
    "--linear",
    help="If probs should be linear. Default is related to cluster.",
    action='store_true')
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
                    if not args.linear:
                        saved_data["cluster_probs"].append(
                            cluster_probs.cpu()[offset].unsqueeze(0))
                        saved_data["cluster_preds"].append(
                            cluster_preds.cpu()[offset].unsqueeze(0))
                        saved_data["img"].append(
                            img.cpu()[offset].unsqueeze(0))
                    else:
                        saved_data["cluster_probs"].append(
                            linear_probs.cpu()[offset].unsqueeze(0))
                        saved_data["cluster_preds"].append(
                            linear_preds.cpu()[offset].unsqueeze(0))
                        saved_data["img"].append(
                            img.cpu()[offset].unsqueeze(0))


def img_unpack(x):
    x = x.unsqueeze(0)
    x_array = np.array(unnorm(x).squeeze(0).cpu())
    I = np.zeros((x_array.shape[2], x_array.shape[2], 3))
    I[:, :, 0] = x_array[0, 0, :, :] * 255  # R
    I[:, :, 1] = x_array[0, 1, :, :] * 255  # G
    I[:, :, 2] = x_array[0, 2, :, :] * 255  # B
    return I.astype(np.uint8)


def correlation_matrix(probs):
    all_points = np.transpose(
        np.reshape(
            probs,
            (probs.shape[1],
             probs.shape[2] *
             probs.shape[3])))
    cor_mat = np.cov(all_points, rowvar=False)
    return cor_mat


def show(saved_data):
    for j in range(len(saved_data["cluster_probs"])):
        probs = np.exp(np.array(saved_data["cluster_probs"][j]))
        n_classes = probs.shape[1]
        step = int(np.sqrt(n_classes))
        x = n_classes // step + 1 * (n_classes % step > 0)
        y = step
        img = img_unpack(saved_data["img"][j])

        # Classes
        f, ax = plt.subplots(x, y)
        plt.suptitle(f"Classes - Element {j}")
        for i in range(n_classes):
            if x == 1 or y == 1:
                ax[i // step].set_axis_off()
                ax[i // step].imshow(img)
                ax[i // step].imshow(probs[0, i, :, :], alpha=0.5,
                                     vmin=0, vmax=np.max(probs[0, i, :, :]))
                ax[i // step].title.set_text(f"Class {i}")

            else:
                ax[i // step, i % step].set_axis_off()
                ax[i // step, i % step].imshow(img)
                ax[i // step,
                   i % step].imshow(probs[0, i, :, :],
                                    alpha=0.5,
                                    vmin=0,
                                    vmax=np.max(probs[0, i, :, :]))
                ax[i // step, i % step].title.set_text(f"Class {i}")

        # Covariance
        if args.covariance:
            cov_mat = correlation_matrix(probs)
            plt.figure(2)
            plt.imshow(cov_mat, cmap="jet")
            plt.title(f"Covariance - Element {j}")

        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()


if __name__ == "__main__":
    model, test_loader, batch_nums, batch_offsets = pre_process()
    evaluate(model, test_loader, batch_nums, batch_offsets)
    shutil.rmtree(model.cfg["experiment_name"])
    show(saved_data)
