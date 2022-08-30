import argparse
import os
import shutil
import sys
import yaml

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cfg",
    "--config",
    help="Configuration file path for the dataset creation, training and evaluation",
    type=str,
    required=True)
args = parser.parse_args()

if __name__ == "__main__":
    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    if params["model_res"] % 8:
        print("Error : Model resolution is not correct. It should be divisible by 8.")
        sys.exit(-1)

    if params["dataset_create_step"]:
        print("PART I : Dataset creation step")

        print("1 - Multispectral images alignement")
        f = os.system(
            f"python ./src/create_dataset1.py -i {params['raw_folder_path']} -o ./temp/part1")
        if f:
            sys.exit(-1)

        print("2 - Linear combination for dimensionality reduction")
        f = os.system(
            f"python ./src/create_dataset2.py -i ./temp/part1 -o ./{params['project_name']}/imgs/train -m {params['LC_matrix_path']} -output_s {params['slice_win_size']} -padding_s {params['slice_pad_step']}")
        if f:
            sys.exit(-1)

        print("3 - Temporary folder deletion")
        f = shutil.rmtree("./temp")
        if f:
            sys.exit(-1)

        print("4 - Output dataset sent to PyTorch folder")
        shutil.move(
            f"./{params['project_name']}",
            "C:/workspace/datadrive/pytorch-data/")

    # Model training
    if params["model_train_step"]:
        print("\nPART II : Model training step")

        print("1 - Training parameters update")
        with open(os.path.join("src", "train_config_template.yml")) as file:
            model_params = yaml.load(file, Loader=yaml.FullLoader)
        model_params["num_workers"] = params["num_workers"]
        model_params["batch_size"] = params["batch_size"]
        model_params["experiment_name"] = params["project_name"]
        model_params["dir_dataset_name"] = params["project_name"]
        model_params["max_epochs"] = params["training_max_epochs"]
        model_params["checkpoint_freq"] = params["training_save_freq"]
        model_params["dir_dataset_n_classes"] = params["model_num_classes"]
        model_params["res"] = params["model_res"]
        with open(os.path.join("..", "src", "configs", "train_config.yml"), 'w') as file:
            yaml.dump(model_params, file)

        print("2 - KNN calculation")
        f = os.system("python ../src/precompute_knns.py")
        if f:
            sys.exit(-1)

        print("3 - STEGO training")
        f = os.system("python ../src/train_segmentation.py")
        if f:
            sys.exit(-1)

        print("4 - Retrieving model")
        model_folder_path = os.path.join(
            "..", "checkpoints", params["project_name"])
        subfolder_path = os.path.join(
            model_folder_path,
            os.listdir(model_folder_path)[0])
        model_path = os.path.join(
            subfolder_path,
            os.listdir(subfolder_path)[0])
        shutil.copy(
            model_path,
            os.path.join(
                "C:/workspace/datadrive/pytorch-data/",
                params["project_name"],
                f"{params['project_name']}_model.ckpt"))
        shutil.rmtree(model_folder_path)

    # Ortho evaluation
    if params["ortho_eval_step"]:
        print("\nPART III : Orthophoto model evaluation step")
        stego_model_path = os.path.join(
            "C:/workspace/datadrive/pytorch-data/",
            params["project_name"],
            f"{params['project_name']}_model.ckpt")
        if params["class_mode"] == "manual":
            print("1 - Test dataset insight")
            if params['linear']:
                f = os.system(
                    f"python ../src/eval_segmentation_simple.py -i {params['test_folder_path']} -m {stego_model_path} -l")
            else:
                f = os.system(
                    f"python ../src/eval_segmentation_simple.py -i {params['test_folder_path']} -m {stego_model_path}")
            if f:
                sys.exit(-1)
            try:
                focus_classes_str = (
                    input("Enter interest class index : "))  # Format : 1,2,8
                focus_classes = focus_classes_str.split(",")
                for i in focus_classes:  # Check number validity
                    a = int(i)
            except BaseException:
                print("Error reading class index.")
                sys.exit(-1)

            print("2 - Orthophoto slicing and processing with linear combination")
            f = os.system(
                f"python ./src/eval_ortho1.py -i_ortho {params['ortho_folder_path']} -m {params['LC_matrix_path']} -o ./temp/part1 -output_s {params['ortho_win_size']} -padding_s {params['ortho_pad_step']}")
            if f:
                sys.exit(-1)

            print("3 - STEGO model evaluation over orthophoto slices")
            if params['linear']:
                f = os.system(
                    f"python ../src/eval_segmentation_class.py -i ./temp/part1 -o ./temp/part2 -c {' '.join(focus_classes)} -m {stego_model_path} -b_size {1} -n_workers {1} -l")
            else:
                f = os.system(
                    f"python ../src/eval_segmentation_class.py -i ./temp/part1 -o ./temp/part2 -c {' '.join(focus_classes)} -m {stego_model_path} -b_size {1} -n_workers {1}")
            if f:
                sys.exit(-1)

            print("4 - Assembling the probability maps slices and exporting")
            f = os.system(
                f"python ./src/eval_ortho2.py -i_ortho {params['ortho_folder_path']} -i_slices ./temp/part2 -o {params['output_folder']} -output_s {params['ortho_win_size']} -dpi {params['ortho_export_dpi']} -n_workers {4*params['num_workers']}")
            if f:
                sys.exit(-1)

            print("5 - Temporary folder deletion")
            f = shutil.rmtree("./temp/")
            if f:
                sys.exit(-1)

        elif params["class_mode"] == "auto":
            print("1 - Orthophoto slicing and processing with linear combination")
            f = os.system(
                f"python ./src/eval_ortho1.py -i_ortho {params['ortho_folder_path']} -m {params['LC_matrix_path']} -o ./temp/part1 -output_s {params['ortho_win_size']} -padding_s {params['ortho_pad_step']}")
            if f:
                sys.exit(-1)
            print("2 - STEGO model evaluation over orthophoto slices")
            if params['linear']:
                f = os.system(
                    f"python ../src/eval_segmentation_class.py -i ./temp/part1 -o ./temp/part2 -c -1 -m {stego_model_path} -b_size {1} -n_workers {1} -l")
            else:
                f = os.system(
                    f"python ../src/eval_segmentation_class.py -i ./temp/part1 -o ./temp/part2 -c -1 -m {stego_model_path} -b_size {1} -n_workers {1}")
            if f:
                sys.exit(-1)

            print("3 - Assembling the probability maps slices and exporting")
            f = os.system(
                f"python ./src/eval_ortho2.py -i_ortho {params['ortho_folder_path']} -i_slices ./temp/part2 -o {params['output_folder']} -output_s {params['ortho_win_size']} -dpi {params['ortho_export_dpi']} -n_workers {4*params['num_workers']}")
            if f:
                sys.exit(-1)

            print("4 - Temporary folder deletion")
            f = shutil.rmtree("./temp/")
            if f:
                sys.exit(-1)
