import os
import logging as log
from typing import Union

import torch

from plenoxels.my_tqdm import tqdm
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.runners.multiscene_trainer import Trainer
from plenoxels.runners.video_trainer import VideoTrainer


@torch.no_grad()
def render_to_path(trainer: Union[VideoTrainer, Trainer], extra_name: str = ""):
    dataset = trainer.test_dataset

    pb = tqdm(total=100, desc=f"Rendering scene")
    frames = []
    for img_idx, data in enumerate(dataset):
        ts_render = trainer.eval_step(data)

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]
        preds_rgb = (
            ts_render["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .byte()
            .numpy()
        )
        frames.append(preds_rgb)
        pb.update(1)
    pb.close()

    out_fname = os.path.join(trainer.log_dir, f"rendering_path_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
