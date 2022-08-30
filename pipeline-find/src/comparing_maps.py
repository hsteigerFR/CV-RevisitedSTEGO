import argparse
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i_select",
    "--input_selection",
    help="Path to the selection maps folder",
    type=str,
    required=True)
parser.add_argument(
    "-i_eval",
    "--input_evaluation",
    help="Path to the evaluation maps folder",
    type=str,
    required=True)
parser.add_argument(
    "-fp_cost",
    "--fp_cost",
    help="False positive weight in the score calculus",
    type=float,
    default=0.2)
parser.add_argument(
    "-d",
    "--display",
    help="Should display superposition maps",
    action="store_true")
args = parser.parse_args()

img_list = [element for element in os.listdir(
    args.input_selection) if os.path.splitext(element)[-1] == ".png"]
K = len([element for element in os.listdir(args.input_evaluation)
        if os.path.isdir(os.path.join(args.input_evaluation, element))])
for img_name in img_list:

    # Importing related pictures
    select_I = cv2.imread(
        os.path.join(
            args.input_selection,
            img_name))[
        :,
        :,
        0] / 255
    eval_I = np.zeros(select_I.shape[:2] + (K,))
    for k in range(K):
        eval_I[:, :, k] = cv2.resize(cv2.imread(os.path.join(args.input_evaluation, str(
            k), img_name))[:, :, 0] / 255, (select_I.shape[1], select_I.shape[0]))

    # Calculating scores
    score_list = []
    I_list = []
    select_I = np.where(select_I == 0, -args.fp_cost, select_I)

    for k in range(K):
        I_list.append(np.multiply(select_I, eval_I[:, :, k]))
        score_list.append(np.sum(np.multiply(select_I, eval_I[:, :, k])))

    # Showing superposition
    if args.display:
        f, ax = plt.subplots(10, 10)
        plt.suptitle("Selection-Eval Scores")
        for k in range(K):
            ax[k // 10, k % 10].imshow(I_list[k])
            ax[k // 10, k % 10].title.set_text(f"{k}")
        plt.show()

    score_dict = {cl: score_list[cl] for cl in np.argsort(score_list)[::-1]}
    print(f"{img_name} : {score_dict}\n\n")
