# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "lego_test",
    "logdir": "./logs",
    "device": "cuda:0",

    # Data settings
    "data_resolution": None,
    "data_downsample": 1,
    "data_dirs": ["/data/DATASETS/SyntheticNerf/lego"],
    #"data_dirs": ["/data/DATASETS/LLFF/fern"],
    # Data settings for 360
    "max_tr_frames": None,
    "max_ts_frames": 10,
    # Data settings for 360
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_steps": 35000,
    "batch_size": 4096,
    "scheduler_type": None,
    "optim_type": "adam",
    "lr": 2e-2,

    "alpha_threshold": 1e-3,


    "plane_tv_weight": 0.000,
    "plane_tv_what": "Gcoords",

    "l1density_weight": 0.000,

    "volume_tv_weight": 0.00,
    "volume_tv_npts": 100,
    "volume_tv_patch_size": 8,
    "volume_tv_what": "Gcoords",

    # Training settings
    "train_fp16": True,
    "save_every": 35000,
    "valid_every": 20000,
    "save_outputs": True,
    "transfer_learning": False,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 1,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 440,  # Used when raymarch_type is 'fixed'
    "spacing_fn": "linear",
    "single_jitter": True,

    # Model settings
    "sh": True,
    "density_threshold": 1e-4,
    "dmask_update": [],
    "upsample_steps": [],
    "upsample_resolution": [],
    "density_multiplier": 1,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 5,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 10,
    },
    {
        "input_coordinate_dim": 5,
        "resolution": [6, 6, 6, 6, 6],
        "feature_dim": 28,
        "init_std": 0.001,
    }
]
"""
}
