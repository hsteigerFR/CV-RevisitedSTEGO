import argparse
import os
import sys
import shutil
import yaml

import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cfg",
    "--config",
    help="Configuration file path for finding the correct class",
    type=str,
    required=True)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

        print("1 - Annotating pictures ...")
        os.makedirs(os.path.join("temp", "part1"), exist_ok=True)
        img_list = [os.path.join(params["test_folder_path"], element) for element in os.listdir(
            params["test_folder_path"]) if os.path.splitext(element)[-1] == ".png"]
        for img_path in tqdm.tqdm(img_list):
            export_path = os.path.join(
                os.path.join(
                    "temp",
                    "part1",
                    f"{os.path.basename(img_path)}"))
            f = os.system(
                f"python ./src/create_mask.py -i {img_path} -o {export_path}")
            if f or not os.listdir(os.path.join("temp", "part1")):
                sys.exit(-1)

        print("2 - Evaluating pictures ...")
        stego_model_path = os.path.join(
            "C:/workspace/datadrive/pytorch-data/",
            params["project_name"],
            f"{params['project_name']}_model.ckpt")
        if params['linear']:
            f = os.system(
                f"python ../src/eval_segmentation_class.py -i {params['test_folder_path']} -o ./temp/part2 -c -1 -m {stego_model_path} -b_size {params['batch_size']} -n_workers {params['num_workers']} -l")
        else:
            f = os.system(
                f"python ../src/eval_segmentation_class.py -i {params['test_folder_path']} -o ./temp/part2 -c -1 -m {stego_model_path} -b_size {params['batch_size']} -n_workers {params['num_workers']}")
        if f:
            sys.exit(-1)

        print("3 - Finding explaining classes ...")
        if params["display"]:
            f = os.system(
                f"python ./src/comparing_maps.py -i_select ./temp/part1 -i_eval ./temp/part2 -d -fp_cost {params['FP_cost']}")
        else:
            f = os.system(
                f"python ./src/comparing_maps.py -i_select ./temp/part1 -i_eval ./temp/part2 -fp_cost {params['FP_cost']}")
        if f:
            sys.exit(-1)

        print("4 - Temporary folder deletion")
        f = shutil.rmtree("./temp")
        if f:
            sys.exit(-1)
