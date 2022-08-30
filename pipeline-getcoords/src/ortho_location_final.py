import argparse
import os
import re

import cv2
import numpy as np
import pandas as pd
import tqdm

from matplotlib import pyplot as plt
from multiprocessing import Pool
from pyproj import Transformer

parser = argparse.ArgumentParser()
parser.add_argument(
    "-ortho_i",
    "--ortho_path",
    help="Path to the RGB orthophoto",
    type=str,
    required=True)
parser.add_argument(
    "-probs_i",
    "--probs_path",
    help="Path to the probability map. The probability map should overlap the RGB orthophoto.",
    type=str,
    required=True)
parser.add_argument(
    "-o",
    "--output_path",
    help="Path to the output folder. It will be created if it does not exist.",
    type=str,
    required=True)

parser.add_argument(
    "-res_f",
    "--resize_factor",
    help="Resize factor for processing.",
    type=float,
    default=1)
parser.add_argument(
    "-prob_t",
    "--thresh_detect",
    help="Probabilities below this threshold will be ignored.",
    type=float,
    default=0.1)
parser.add_argument(
    "-dist_t",
    "--thresh_dist_pix",
    help="Distance in pixel below which two clusters will merge.",
    type=float,
    default=70)
parser.add_argument(
    "-pwin_s",
    "--process_size",
    help="Size of the processing sliding window.",
    type=float,
    default=300)
parser.add_argument(
    "-owin_s",
    "--outimg_size",
    help="Size of exported anomaly images",
    type=float,
    default=600)
parser.add_argument(
    "-num_w",
    "--num_workers",
    help="Number of workers for multiprocessing.",
    type=float,
    default=15)
parser.add_argument(
    "-max_k",
    "--max_k",
    help="Maximum number of clusters per processing window for KMeans",
    type=float,
    default=15)
parser.add_argument(
    "-max_it",
    "--max_it",
    help="Maximum number of iterations for the weighted KMeans algorithm",
    type=float,
    default=10)
args = parser.parse_args()

resize_f = args.resize_factor
thresh_detect = args.thresh_detect
thresh_dist_pix = resize_f * args.thresh_dist_pix
process_size = args.process_size
outimg_size = args.outimg_size
num_workers = args.num_workers
max_k = args.max_k
max_it = args.max_it


def weighted_k_means(array, n, max_it):
    global thresh_detect
    score_list = [np.nan]
    # 1) Get pixels in image where > 0.1
    H, W = array.shape
    points = []
    for u in range(W):
        for v in range(H):
            if array[v, u] > thresh_detect:
                points.append((u, v))

    # 2) Init : Randomly place n means
    means = []
    for i in range(n):
        rand_index = np.random.randint(0, len(points))
        means.append(points[rand_index])

    for k in range(max_it):

        # 3) For each pixel, calculate distance and add to closest group
        G = [set() for i in range(n)]
        for point in points:
            dist_list = [
                np.linalg.norm(
                    np.array(point) -
                    np.array(
                        means[i])) for i in range(n)]
            min_index = dist_list.index(min(dist_list))
            G[min_index].add(point)

        # 4) Calculate new group weighted mean
        means = []
        for i in range(n):
            loc_mean = np.zeros((2,))
            loc_norm = 0
            for point in G[i]:
                u, v = point
                loc_mean = loc_mean + array[v, u] * np.array(point)
                loc_norm += array[v, u]
            if loc_norm > 0:
                loc_mean = loc_mean / loc_norm
            else:
                loc_mean = np.zeros(2)
            means.append(loc_mean)

        # 5) Calculate score
        score = 0
        for i in range(n):
            for point in G[i]:
                u, v = point
                score += array[v, u] * \
                    np.linalg.norm(np.array(point) - np.array(means[i]))
        score_list.append(score)
        if len(score_list) > 1 and score_list[-1] == score_list[-2]:
            return G, means

    return G, means


def cluster_fusion(G, means):
    global thresh_dist_pix
    M = np.zeros((len(means), len(means)))
    for i in range(len(means)):
        for j in range(len(means)):
            M[i, j] = np.linalg.norm(means[i] - means[j])

    browsing_indexes = [i for i in range(len(means))]
    dict_res = {}

    while browsing_indexes:
        index1 = browsing_indexes[0]
        L = []
        for index2 in browsing_indexes:
            if index1 == index2 or M[index1, index2] < thresh_dist_pix:
                L.append(index2)
        dict_res[index1] = L.copy()
        for index in dict_res[index1]:
            browsing_indexes.remove(index)

    final_means = []
    for key in dict_res.keys():
        indexes = dict_res[key]
        norm_factor = np.sum([len(G[index]) for index in indexes])
        if norm_factor:
            corresponding_means = [
                len(G[index]) * means[index] / norm_factor for index in indexes]
            final_means.append(np.sum(corresponding_means, axis=0))
    return final_means


def get_coordinates(u, v, upper_left_tuple, pix_size_tuple, resize_f):
    transformer = Transformer.from_crs(32631, 4326)  # UTM31N to WGS84
    res_UTMN31 = np.array(upper_left_tuple) + np.array(
        [u * pix_size_tuple[0] / resize_f, v * pix_size_tuple[1] / resize_f])
    res_WGS84 = transformer.transform(res_UTMN31[0], res_UTMN31[1])
    return res_WGS84


def plot_coordinates(lat, lon, upper_left_tuple, pix_size_tuple, resize_f):
    transformer = Transformer.from_crs(4326, 32631)  # WGS84 to UTM31N
    (X, Y) = transformer.transform(lat, lon)
    (Ox, Oy) = upper_left_tuple
    pix_size = pix_size_tuple[0] / resize_f
    u = (X - Ox) / pix_size
    v = -(Y - Oy) / pix_size
    return (u, v)


def create_anomaly_image(ortho, u, v, img_size, index):
    img = ortho[int(v - img_size / 2):int(v + img_size / 2),
                int(u - img_size / 2):int(u + img_size / 2), :].copy()
    start_point = (int(img_size / 2 - img_size / 12),
                   int(img_size / 2 - img_size / 10))
    end_point = (int(img_size / 2 + img_size / 12),
                 int(img_size / 2 + img_size / 10))
    img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    img = cv2.putText(img, f"id:{index}", (10, 30),
                      font, fontScale, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def pre_process():
    print("Creating output directory...")
    os.makedirs(args.output_path, exist_ok=True)

    print("Importing input files...")
    ortho = cv2.imread(args.ortho_path)
    H, W, _ = ortho.shape
    probs = cv2.imread(args.probs_path)[:H, :W, 0] / 255

    H, W = int(resize_f * H), int(resize_f * W)
    ortho = cv2.resize(ortho, (W, H))
    probs = cv2.resize(probs, (W, H))

    print("Getting ortho info...")
    upper_left_str = os.popen(
        f"gdalinfo -mdd xml:XMP {args.ortho_path} | findstr \"Upper Left\"").readlines()[0]
    pix_size_str = os.popen(
        f"gdalinfo -mdd xml:XMP {args.ortho_path} | findstr \"Pixel Size\"").readlines()[1]
    upper_left_tuple = tuple(map(float, re.findall(
        r'\((.*?,.*?)\)', upper_left_str)[0].split(",")))
    pix_size_tuple = tuple(map(float, re.findall(
        r'\((.*?,.*?)\)', pix_size_str)[0].split(",")))

    print("Preprocessing data ...")
    to_process = []
    h = w = process_size
    key_points = []
    for v in tqdm.tqdm(range(0, ortho.shape[0], h)):
        for u in range(0, ortho.shape[1], w):
            ortho_part = ortho[v:v + h, u:u + w, :]
            if ortho_part.any():
                y_pred = probs[v:v + h, u:u + w]
                to_process.append((y_pred.copy(), u, v))
    return to_process, ortho, (upper_left_tuple, pix_size_tuple)


def process(element):
    loc_detect = []
    y_pred, u, v = element
    if (y_pred > thresh_detect).any():

        # Modified KMeans
        G, means = weighted_k_means(y_pred, max_k, max_it)
        final_means = cluster_fusion(G, means)

        # Saving Keypoints
        for coord in final_means:
            loc_u, loc_v = coord
            loc_detect.append((loc_u + u, loc_v + v))
    return loc_detect


def show_and_export(key_points, ortho_params):
    upper_left_tuple = ortho_params[0]
    pix_size_tuple = ortho_params[1]
    print("Displaying Results...")
    plt.imshow(cv2.cvtColor(ortho, cv2.COLOR_BGR2RGB))
    for i, coord in enumerate(key_points):
        u, v = coord
        lat, lon = get_coordinates(
            u, v, upper_left_tuple, pix_size_tuple, resize_f)
        plt.scatter(u, v, c="red")
        plt.text(u, v, f"C{i}")
    plt.show()

    print("Exporting results...")
    table = pd.DataFrame(columns=["id", "lat", "lon"])
    for i, coord in enumerate(key_points):
        u, v = coord
        anomaly_ret = create_anomaly_image(ortho, u, v, process_size, i)
        if anomaly_ret.size != 0:
            img = cv2.resize(anomaly_ret, (outimg_size, outimg_size))
            cv2.imwrite(os.path.join(args.output_path, f"{i}.png"), img)
            lat, lon = get_coordinates(
                u, v, upper_left_tuple, pix_size_tuple, resize_f)
            table = pd.concat([table, pd.DataFrame.from_records(
                [{"id": i, "lat": lat, "lon": lon}])])
    table.to_csv(os.path.join(args.output_path, "export.csv"), index=False)


if __name__ == "__main__":
    to_process, ortho, ortho_params = pre_process()
    print("Processing data...")
    with Pool(num_workers) as p:
        r = list(tqdm.tqdm(p.imap(process, to_process), total=len(to_process)))
    key_points = [item for sublist in r for item in sublist]
    print(f"{len(key_points)} anomalies detected")
    show_and_export(key_points, ortho_params)
