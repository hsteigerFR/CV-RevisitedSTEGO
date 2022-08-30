import argparse
import os


import cv2
import numpy as np
import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i_ortho",
    "--ortho_path",
    help="Path to the orthophoto bands folder",
    type=str,
    required=True)
parser.add_argument(
    "-i_slices",
    "--slices_path",
    help="Path to the folder containing probability map slices",
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
    default=400)
parser.add_argument(
    "-n_workers",
    "--num_workers",
    help="Number of multiprocessign workers",
    type=int,
    default=4)
parser.add_argument(
    "-dpi",
    "--export_dpi",
    help="Resolution of the exported superposition map.",
    type=int,
    default=400)
args = parser.parse_args()


def prepare():
    print("Creating output folder ...")
    os.makedirs(args.output_path, exist_ok=True)

    print("Importing RGB orthophoto ...")
    L = [os.path.join(args.ortho_path, element) for element in os.listdir(
        args.ortho_path) if os.path.splitext(element)[-1] == ".tif"]
    ortho_init_path = ""
    for element in L:
        if "transparent_mosaic_group1" in element:
            ortho_init_path = element
    return ortho_init_path


def process(element):
    focus_class, ortho_init_path = element
    ortho_init = cv2.cvtColor(cv2.imread(ortho_init_path), cv2.COLOR_BGR2RGB)
    H = ortho_init.shape[0] + 200
    W = ortho_init.shape[1] + 200
    class_folder_path = os.path.join(args.slices_path, focus_class)
    L = [os.path.join(class_folder_path, element) for element in os.listdir(
        class_folder_path) if os.path.splitext(element)[-1] == ".png"]
    win_size = args.output_size

    array = np.zeros((H, W))
    norm_array = np.zeros((H, W))
    for img_path in L:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        L = img_name.split("_")
        u = int(L[0])
        v = int(L[1])

        loc_img = cv2.imread(img_path)[:, :, 0]
        array[v:v + win_size, u:u +
              win_size] += cv2.resize(loc_img, (win_size, win_size)) / 255
        norm_array[v:v + win_size, u:u + win_size] += 1

    array = np.divide(
        array,
        norm_array,
        out=np.zeros_like(array),
        where=norm_array != 0)
    array = (255 * (array - np.min(array)) /
             (np.max(array) - np.min(array))).astype(np.uint8)

    f = plt.figure()
    plt.imshow(ortho_init)
    plt.imshow(array, cmap="jet", alpha=0.5)
    plt.savefig(
        os.path.join(
            args.output_path,
            f"smap_{focus_class}.png"),
        bbox_inches='tight',
        dpi=args.export_dpi)
    cv2.imwrite(
        os.path.join(
            args.output_path,
            f"pmap_{focus_class}.png"),
        array)
    plt.close(f)


if __name__ == "__main__":
    ortho_init_path = prepare()
    print("Assembling evaluation orthomap / saving visualization ...")
    L_classes = [
        (folder_name,
         ortho_init_path) for folder_name in os.listdir(
            args.slices_path) if os.path.isdir(
            os.path.join(
                args.slices_path,
                folder_name))]
    with Pool(args.num_workers) as p:
        r = list(tqdm.tqdm(p.imap(process, L_classes), total=len(L_classes)))
