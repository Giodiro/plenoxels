# configuration file to be used with `run_multi_scene.py`
# the configuration must be specified in a dictionary called `config`.
import numpy as np

samples_per_voxel = 6
resolution = 128
config = {
    "expname": "test",
    "data_resolution": 800,
    "data_downsample": 1,
    "data_dirs": ["/data/datasets/nerf/data/nerf_synthetic/lego"],

    "max_tr_frames": None,
    "max_ts_frames": 10,
    "batch_size": 2000,
    "num_batches_per_dset": 1,
    "num_epochs": 10,
    "scheduler_type": None,
    "optim_type": "adam",
    "model_type": "learnable_hash",
    "logdir": "./logs",
    "train_fp16": True,
    "save_every": 1,
    "valid_every": 1,
    "transfer_learning": False,

    "lr": 2e-3,

    "n_intersections": resolution * samples_per_voxel,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 2,
        "resolution": 128,
        "rank": 20,
        "init_std": 0.1,
    },
    {
        "input_coordinate_dim": 4,
        "resolution": 8,
        "feature_dim": 32,
        "init_std": 0.05
    }
]
"""
}
