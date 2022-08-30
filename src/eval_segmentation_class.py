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
    "-o",
    "--output",
    help="Path to output folder",
    type=str,
    required="True")
parser.add_argument(
    "-c",
    "--eval_class",
    help="Class(es) to evaluate",
    type=int,
    nargs='+',
    required="True")
parser.add_argument(
    "-m",
    "--model",
    help="Path to model .ckpt file",
    type=str,
    required="True")
parser.add_argument(
    "-n_workers",
    "--num_workers",
    help="Number of processes used for evaluation",
    type=int,
    default=10)
parser.add_argument(
    "-b_size",
    "--batch_size",
    help="Batch size for evaluation",
    type=int,
    default=10)
parser.add_argument(
    "-l",
    "--linear",
    help="If probs should be linear. Default is related to cluster.",
    action='store_true')
args = parser.parse_args()


def map2image(array, norm_min, norm_max):
    return (255 * (array - norm_min) / (norm_max - norm_min)).astype(np.uint8)


def pre_process():
    print("Getting model ...")
    model = LitUnsupervisedSegmenter.load_from_checkpoint(args.model)
    res = model.cfg["res"]
    model_name = model.cfg["experiment_name"]

    print("Creating output folder ...")
    os.makedirs(args.output, exist_ok=True)

    print("Generating dataset ...")
    filenames = []
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
            filenames.append(image_name)
            shutil.copy(image_path, os.path.join(loc_path, image_name))
    else:
        image_name = os.path.basename(args.input)
        filenames = [image_name]
        shutil.copy(args.input, os.path.join(loc_path, image_name))

    print("Importing dataset ...")
    batch_size = args.batch_size
    num_workers = args.num_workers
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

    return model, test_loader, batch_nums, batch_offsets, filenames


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


if __name__ == "__main__":
    model, test_loader, batch_nums, batch_offsets, filenames = pre_process()
    evaluate(model, test_loader, batch_nums, batch_offsets)
    shutil.rmtree(model.cfg["experiment_name"])

    print("Getting normalization parameters ...")
    min_L = {eval_class: []
             for eval_class in range(saved_data["cluster_probs"][0].shape[1])}
    max_L = {eval_class: []
             for eval_class in range(saved_data["cluster_probs"][0].shape[1])}
    for i in tqdm(range(len(saved_data["cluster_probs"]))):
        probs = np.exp(np.array(saved_data["cluster_probs"][i]))

        if args.eval_class == [-1]:  # Export all classes
            for eval_class in range(0, probs.shape[1]):
                min_L[eval_class].append(np.min(probs[0, eval_class, :, :]))
                max_L[eval_class].append(np.max(probs[0, eval_class, :, :]))
        else:
            for eval_class in args.eval_class:
                min_L[eval_class].append(np.min(probs[0, eval_class, :, :]))
                max_L[eval_class].append(np.max(probs[0, eval_class, :, :]))

    min_dict = {eval_class: 0 for eval_class in range(
        saved_data["cluster_probs"][0].shape[1])}
    max_dict = {eval_class: 1 for eval_class in range(
        saved_data["cluster_probs"][0].shape[1])}
    for i in range(len(saved_data["cluster_probs"])):
        if args.eval_class == [-1]:  # Export all classes
            for eval_class in range(0, probs.shape[1]):
                min_dict[eval_class] = min(min_L[eval_class])
                max_dict[eval_class] = max(max_L[eval_class])
        else:
            for eval_class in args.eval_class:
                min_dict[eval_class] = min(min_L[eval_class])
                max_dict[eval_class] = max(max_L[eval_class])

    print("Creating output files ...")
    filenames = sorted(filenames)
    for i in tqdm(range(len(saved_data["cluster_probs"]))):
        probs = np.exp(np.array(saved_data["cluster_probs"][i]))

        if args.eval_class == [-1]:  # Export all classes
            for eval_class in range(0, probs.shape[1]):
                os.makedirs(
                    os.path.join(
                        args.output,
                        str(eval_class)),
                    exist_ok=True)
                export_path = os.path.join(
                    args.output, str(eval_class), filenames[i])
                export_array = map2image(
                    probs[0, eval_class, :, :], min_dict[eval_class], max_dict[eval_class])
                cv2.imwrite(export_path, export_array)
        else:
            for eval_class in args.eval_class:
                os.makedirs(
                    os.path.join(
                        args.output,
                        str(eval_class)),
                    exist_ok=True)
                export_path = os.path.join(
                    args.output, str(eval_class), filenames[i])
                export_array = map2image(
                    probs[0, eval_class, :, :], min_dict[eval_class], max_dict[eval_class])
                cv2.imwrite(export_path, export_array)
