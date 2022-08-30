import os
import shutil
import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt

from modules import *
from data import *
from collections import defaultdict
import torch.multiprocessing
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
    "-c",
    "--classes",
    nargs=2,
    help="Classes for covariance calculus",
    type=int,
    required="True")
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


def correlation_matrix(probs, classes):
    all_points = np.transpose(
        np.reshape(
            probs,
            (probs.shape[1],
             probs.shape[2] *
             probs.shape[3])))
    cor_mat = np.cov(all_points, rowvar=False)
    """
    reduced_cor_mat = np.zeros((2,2))
    index_i = classes[0]
    index_j = classes[1]
    reduced_cor_mat[0,0] = cor_mat[index_i,index_i]
    reduced_cor_mat[1,1] = cor_mat[index_j,index_j]
    reduced_cor_mat[0,1] = reduced_cor_mat[1,0] = cor_mat[index_i,index_j]
    """
    return cor_mat


def correlation_map(probs, classes):
    cor_map = np.zeros((probs.shape[2], probs.shape[3]))
    index_i = classes[0]
    index_j = classes[1]

    map_i = probs[0, index_i, :, :]
    map_j = probs[0, index_j, :, :]
    mean_i = np.mean(map_i)
    mean_j = np.mean(map_j)

    cor_map = np.multiply((map_i - mean_i),
                          (map_j - mean_j)) / (mean_j * mean_i)
    return cor_map


def correlation_map_onevsall(probs, my_class):
    cor_map_all = np.zeros((probs.shape[2], probs.shape[3]))
    maps = [probs[0, i, :, :] for i in range(probs.shape[1])]
    means = [np.mean(maps[i]) for i in range(len(maps))]
    for i in range(len(maps)):
        if i != my_class:
            cor_map_all += np.multiply((maps[i] - means[i]),
                                       (maps[my_class] - means[my_class])) / (means[i] * means[my_class])
    return cor_map_all


def correlation_map_overall(probs):
    cor_map_all = np.zeros((probs.shape[2], probs.shape[3]))
    maps = [probs[0, i, :, :] for i in range(probs.shape[1])]
    means = [np.mean(maps[i]) for i in range(len(maps))]

    for i in range(len(maps)):
        for j in range(len(maps)):
            if i != j:
                cor_map_all += np.multiply((maps[i] - means[i]),
                                           (maps[j] - means[j])) / (means[i] * means[j])
    return cor_map_all


def show(saved_data, classes):
    for i in range(len(saved_data["cluster_probs"])):
        probs = np.exp(np.array(saved_data["cluster_probs"][i]))
        img = img_unpack(saved_data["img"][i])

        index_i = classes[0]
        index_j = classes[1]

        f, ax = plt.subplots(2, 4)
        plt.suptitle(f"Image {i}")

        ax[0, 0].imshow(img)
        ax[0, 0].imshow(probs[0, index_i, :, :], alpha=0.7)
        ax[0, 0].title.set_text(f"Probs C{index_i}")

        ax[0, 1].imshow(img)
        ax[0, 1].imshow(probs[0, index_j, :, :], alpha=0.7)
        ax[0, 1].title.set_text(f"Probs C{index_j}")

        cor_map = correlation_map(probs, classes)
        ax[0, 2].imshow(img)
        ax[0, 2].imshow(cor_map, cmap="jet", alpha=0.7)
        ax[0, 2].title.set_text(f"Correlation Map C{index_i}/C{index_j}")

        cor_map_overall = correlation_map_overall(probs)
        ax[0, 3].imshow(img)
        ax[0, 3].imshow(cor_map_overall, cmap="jet", alpha=0.7)
        ax[0, 3].title.set_text("Overall Correlation Map")

        cor_map_ivsall = correlation_map_onevsall(probs, index_i)
        ax[1, 0].imshow(img)
        ax[1, 0].imshow(cor_map_ivsall, cmap="jet", alpha=0.7)
        ax[1, 0].title.set_text(f"Correlation Map C{index_i}/All")

        cor_map_jvsall = correlation_map_onevsall(probs, index_j)
        ax[1, 1].imshow(img)
        ax[1, 1].imshow(cor_map_jvsall, cmap="jet", alpha=0.7)
        ax[1, 1].title.set_text(f"Correlation Map C{index_j}/All")

        cor_mat = correlation_matrix(probs, classes)
        ax[1, 2].imshow(cor_mat, cmap="jet")
        ax[1, 2].title.set_text(f"Correlation Matrix")

        plt.show()


if __name__ == "__main__":
    model, test_loader, batch_nums, batch_offsets = pre_process()
    evaluate(model, test_loader, batch_nums, batch_offsets)
    shutil.rmtree(model.cfg["experiment_name"])
    show(saved_data, args.classes)
