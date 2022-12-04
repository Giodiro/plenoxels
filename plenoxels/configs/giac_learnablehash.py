# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
import numpy as np
config = {
    "expname": "fern",
    "logdir": "./logs",
    "device": "cuda:0",
    "wandb": False,

    # Data settings
    "data_downsample": 4,
    #"data_dirs": ["/data/DATASETS/SyntheticNerf/ficus"],
    "data_dirs": ["/data/DATASETS/LLFF/fern"],
    # Data settings for 360
    "max_tr_frames": None,
    "max_ts_frames": 50,
    # Data settings for 360
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_steps": 40001,
    "scheduler_type": "warmup_cosine",
    "lr": 2e-2,
    "cone_angle": 0.0,
    "optim_type": "adam",

    "plane_tv_weight": 0.0004,
    "plane_tv_what": "Gcoords",
    "l1density_weight": 0.000,
    "binary_reg_weight": 0.0,
    "distortion_loss_weight": 0.0,

    # Training settings
    "train_fp16": True,
    "save_every": 10000,
    "valid_every": 10000,
    "save_outputs": True,
    "transfer_learning": False,

    # Raymarching settings
    "sample_batch_size": 1 << 18,
    "n_samples": 1024,
    "early_stop_eps": 1e-4,

    # Model settings
    "sh": False,
    "use_F": False,
    "density_activation": "trunc_exp",
    "alpha_threshold": 1e-3,
    "density_threshold": 1e-2,
    "occupancy_grid_resolution": [256, 256, 256],
    "multiscale_res": [1.44727 ** i for i in range(8)],
    "concat_features": True,
    "train_every_scale": False,

    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 2,
        "resolution": [32, 32, 32],
    },
]
"""
}
