import argparse
import os
import shutil
import sys
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cfg",
    "--config",
    help="Configuration file path for the algorithm",
    type=str,
    required=True)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    f = os.system(
        f"python ./src/ortho_location_final.py -ortho_i {params['ortho_path']} -probs_i {params['prob_map_path']} -o {params['output_folder_path']} -res_f {params['processing_resize_factor']} -prob_t {params['probs_threshold']} -dist_t {params['cluster_separation_pix']}")
