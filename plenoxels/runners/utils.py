from typing import Union
import logging as log

import numpy as np

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.runners.multiscene_trainer import Trainer
from plenoxels.runners.phototourism_trainer import PhototourismTrainer
from plenoxels.runners.video_trainer import VideoTrainer

__all__ = (
    "initialize_model",
)


def initialize_model(runner: Union[Trainer, PhototourismTrainer, VideoTrainer], **kwargs):
    dset = runner.test_dataset
    try:
        global_translation = dset.global_translation
    except AttributeError:
        global_translation = None
    try:
        global_scale = dset.global_scale
    except AttributeError:
        global_scale = None
    num_images = None
    if runner.train_dataset is not None:
        try:
            num_images = runner.train_dataset.num_images
        except AttributeError:
            num_images = None
    model = LowrankModel(
        grid_config=kwargs.pop("grid_config"),
        aabb=dset.scene_bbox,
        is_ndc=dset.is_ndc,
        is_contracted=dset.is_contracted,
        global_scale=global_scale,
        global_translation=global_translation,
        use_appearance_embedding=isinstance(runner, PhototourismTrainer),
        num_images=num_images,
        **kwargs)
    log.info(f"Initialized {model.__class__} model with "
             f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, "
             f"using ndc {model.is_ndc} and contraction {model.is_contracted}. "
             f"Linear decoder: {model.linear_decoder}.")
    return model

