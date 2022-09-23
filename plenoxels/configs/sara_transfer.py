# configuration file to be used with `main.py` for transfer learning experiments
# the configuration must be specified in a dictionary called `config`.
import numpy as np
config = {
    "expname": "allbutlego128transfer",
    "logdir": "./logs",

    # Data settings
    "data_resolution": None,
    "data_downsample": 4,
    "data_dirs": [
        # "/home/sfk/data/nerf_synthetic/chair",
        # "/home/sfk/data/nerf_synthetic/drums",
        # "/home/sfk/data/nerf_synthetic/ficus",
        # "/home/sfk/data/nerf_synthetic/hotdog",
        "/home/sfk/data/nerf_synthetic/lego",
        # "/home/sfk/data/nerf_synthetic/materials",
        # "/home/sfk/data/nerf_synthetic/mic",
        # "/home/sfk/data/nerf_synthetic/ship",
        ],
    # Data settings for 360
    "max_tr_frames": None,
    "max_ts_frames": None,
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_epochs": 10,
    "batch_size": 4096,
    "num_batches_per_dset": 10,
    "scheduler_type": None,
    "optim_type": "adam",
    "lr": 2e-3,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0,
    "regnerf_weight_max_step": 512,
    "l1density_weight": 0,
    "plane_tv_weight": 0.0,

    # Training settings
    "train_fp16": True,
    "save_every": 5,
    "valid_every": 5,
    "save_outputs": True,
    "transfer_learning": True,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 2,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 440,  # Used when raymarch_type is 'fixed'
    "spacing_fn": "linear",
    "single_jitter": True,

    # Model settings
    "density_threshold": 1e-4,
    "dmask_update": [np.inf],
    "density_multiplier": 1,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 5,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 10,
        "init_std": 0.2,  
    },
    {
        "input_coordinate_dim": 5,
        "resolution": [4, 4, 4, 4, 4],
        "feature_dim": 32,
        "init_std": 0.05
    }
]
"""
}
