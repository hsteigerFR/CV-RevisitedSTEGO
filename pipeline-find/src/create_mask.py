import argparse
import os
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

all_polys = []
poly_points = []
I = 0
f = 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input_path",
    help="Path to the input picture",
    type=str,
    required=True)
parser.add_argument(
    "-o",
    "--output_path",
    help="Path to the output file",
    type=str,
    required=True)
args = parser.parse_args()


def onclick(event):
    global poly_points
    global all_polys
    global I
    global f

    if event.button == MouseButton.LEFT:
        x, y = event.xdata, event.ydata
        poly_points.append((x, y))
        if len(poly_points) > 1:
            plt.plot([poly_points[-2][0], x], [poly_points[-2][1], y], c="red")
            if len(poly_points) > 2 and np.linalg.norm(np.array(
                    poly_points[0]) - np.array(poly_points[-1]), 2) < 0.01 * np.min(I.shape[:2]):
                all_polys.append(poly_points.copy())
                poly_points = []
        plt.scatter(x, y, c="red")
        plt.draw()

    elif event.button == MouseButton.RIGHT:
        if poly_points != []:
            poly_points = []
        else:
            if all_polys:
                all_polys.pop()
        plt.clf()
        plt.imshow(I)
        for poly in all_polys:
            for i in range(len(poly)):
                plt.scatter(poly[i][0], poly[i][1], c="red")
                if i < len(poly) - 1:
                    plt.plot([poly[i][0], poly[i + 1][0]],
                             [poly[i][1], poly[i + 1][1]], c="red")
            plt.plot([poly[0][0], poly[-1][0]],
                     [poly[0][1], poly[-1][1]], c="red")
        plt.draw()

    elif event.button == MouseButton.MIDDLE:
        if poly_points == [] and all_polys != []:
            plt.close(f)
            process_data()


def process_data():
    global all_polys
    global I

    if not all_polys:
        sys.exit(-1)

    # Processing result
    res = np.zeros(I.shape[:2])
    for poly in all_polys:
        poly_shape = Polygon(poly)
        box = poly_shape.minimum_rotated_rectangle
        X, Y = box.exterior.coords.xy
        for u in range(int(np.min(X)), int(np.max(X))):
            for v in range(int(np.min(Y)), int(np.max(Y))):
                x, y = u, v
                point = Point(x, y)
                if poly_shape.contains(point):
                    res[v, u] = 255
    cv2.imwrite(args.output_path, res)


def start_selection_window():
    global I
    global f

    # Opening picture
    I = cv2.cvtColor(cv2.imread(args.input_path), cv2.COLOR_BGR2RGB)

    # Opening selection window
    f = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.state('zoomed')
    cid = f.canvas.mpl_connect('button_press_event', onclick)

    plt.imshow(I)
    plt.show()


if __name__ == "__main__":
    start_selection_window()
