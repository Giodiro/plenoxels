import os
import logging as log
from typing import Union

import torch

from plenoxels.my_tqdm import tqdm
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.runners.multiscene_trainer import Trainer
from plenoxels.runners.video_trainer import VideoTrainer
from plenoxels.raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)


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

def normalize_for_disp(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img

@torch.no_grad()
def decompose_space_time(trainer: Trainer, extra_name: str = ""):
    # store original parameters
    parameters = []
    for multires_grids in trainer.model.field.grids:
        parameters.append([grid.data for grid in multires_grids])
    # TODO: store density model params

    dataset = trainer.test_dataset
    
    frames = []
    for img_idx, data in enumerate(dataset):
        if img_idx == 15:
            camdata = data
    num_frames = img_idx + 1
    pb = tqdm(total=num_frames, desc=f"Rendering scene with separate space and time components")

    for img_idx, _ in enumerate(dataset):
        camdata["timestamps"] = torch.Tensor([float(img_idx) / num_frames])

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]

        # Full model
        for i in range(len(trainer.model.field.grids)):
            # for plane_idx in [0, 1, 3]:  # space-grids on
            #     trainer.model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
            for plane_idx in [2, 4, 5]:  # time-grids on
                trainer.model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        # TODO: do the same for the proposal models
        # for density_model in trainer.model.proposal_networks:
        #     for i in range(len(density_model.grids)):
        #     # for plane_idx in [0, 1, 3]:  # space-grids on
        #     #     trainer.model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        #     for plane_idx in [2, 4, 5]:  # time-grids on
        #         density_model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        # trainer.model.density_fns.extend([network.get_density for network in self.proposal_networks])

        preds = trainer.eval_step(camdata)
        full = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        
        # Model without time grids
        for i in range(len(trainer.model.field.grids)):    
            for plane_idx in [2, 4, 5]:  # time-grids off
                trainer.model.field.grids[i][plane_idx].data = torch.ones_like(parameters[i][plane_idx])
            # for plane_idx in [0, 1, 3]:  # space-grids on
            #     trainer.model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        # TODO: do the same for the proposal models
        preds = trainer.eval_step(camdata)
        spatial = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Time
        temporal = normalize_for_disp(full - spatial)

        frame = torch.cat([full, spatial, temporal], dim=1).clamp(0, 1).mul(255.0).byte().numpy()
        frames.append(frame)
        pb.update(1)
    pb.close()

    out_fname = os.path.join(trainer.log_dir, f"spacetime_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
