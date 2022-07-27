from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter

torch.multiprocessing.set_sharing_strategy('file_system')


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

        loader_crop = "center"
        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name="directory",
            crop_type=None,
            image_set="train",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            cfg=model.cfg,
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size,
                                 shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True, collate_fn=flexible_collate)

        model.eval().cuda()

        par_model = model.net

        all_good_images = [i for i in range(cfg.start, cfg.end)]
        batch_nums = torch.tensor([n // (cfg.batch_size)
                                  for n in all_good_images])
        batch_offsets = torch.tensor(
            [n % (cfg.batch_size) for n in all_good_images])

        saved_data = defaultdict(list)
        with Pool(cfg.num_workers) as pool:
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

                    linear_probs = torch.log_softmax(
                        model.linear_probe(code), dim=1)
                    cluster_probs = model.cluster_probe(
                        code, 2, log_probs=True)

                    linear_preds = linear_probs.argmax(1)
                    cluster_preds = cluster_probs.argmax(1)

                    model.test_linear_metrics.update(linear_preds, label)
                    model.test_cluster_metrics.update(cluster_preds, label)

                    if i in batch_nums:
                        matching_offsets = batch_offsets[torch.where(
                            batch_nums == i)]
                        for offset in matching_offsets:
                            saved_data["linear_preds"].append(
                                linear_preds.cpu()[offset].unsqueeze(0))
                            saved_data["cluster_preds"].append(
                                cluster_preds.cpu()[offset].unsqueeze(0))
                            saved_data["label"].append(
                                label.cpu()[offset].unsqueeze(0))
                            saved_data["img"].append(
                                img.cpu()[offset].unsqueeze(0))
        saved_data = {k: torch.cat(v, dim=0) for k, v in saved_data.items()}

        tb_metrics = {
            **model.test_linear_metrics.compute(),
            **model.test_cluster_metrics.compute(),
        }

        n_rows = 2
        plt.style.use('dark_background')

        for good_images in batch_list(range(len(all_good_images)), 10):
            fig, ax = plt.subplots(
                n_rows, len(good_images), figsize=(
                    len(good_images) * 3, n_rows * 3))
            for i, img_num in enumerate(good_images):
                plot_img = (
                    prep_for_plot(
                        saved_data["img"][img_num]) *
                    255).numpy().astype(
                    np.uint8)
                Image.fromarray(plot_img).save(
                    join(join(result_dir, "img", str(img_num) + ".jpg")))

                ax[0, i].imshow(plot_img)
                plot_cluster = (model.label_cmap[
                    model.test_cluster_metrics.map_clusters(
                        saved_data["cluster_preds"][img_num])]) \
                    .astype(np.uint8)
                Image.fromarray(plot_cluster).save(
                    join(join(result_dir, "cluster", str(img_num) + ".png")))
                ax[1, i].imshow(plot_cluster)

            ax[0, 0].set_ylabel("Image", fontsize=26)
            ax[1, 0].set_ylabel("STEGO\n(Ours)", fontsize=26)

            remove_axes(ax)
            plt.tight_layout()
            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    prep_args()
    my_app()
