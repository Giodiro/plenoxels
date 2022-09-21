# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "lego_test",
    "logdir": "./logs",

    # Data settings
    "data_resolution": None,
    "data_downsample": 8,
    #"data_dirs": ["/data/DATASETS/SyntheticNerf/lego"],
    "data_dirs": ["/data/DATASETS/LLFF/fern"],
    # Data settings for 360
    "max_tr_frames": None,
    "max_ts_frames": 10,
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_epochs": 10,
    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": None,
    "optim_type": "adam",
    "lr": 2e-3,
    "regnerf_weight_start": 0.0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 700,

    "plane_tv_weight": 0.0,
    "l1density_weight": 0.0,

    # Training settings
    "train_fp16": True,
    "save_every": 10,
    "valid_every": 1,
    "save_outputs": True,
    "transfer_learning": False,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 2,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 440,  # Used when raymarch_type is 'fixed'
    "spacing_fn": "linear",
    "single_jitter": True,

    # Model settings
    "density_threshold": 1e-4,
    "dmask_update": [800, 6000, 12000],
    "density_multiplier": 1,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 5,
        "grid_dimensions": 3,
        "resolution": [282, 314, 188],
        "rank": 1,
        "init_std": 0.01,
    },
    {
        "input_coordinate_dim": 5,
        "resolution": [4, 4, 4, 4, 4],
        "feature_dim": 32,
        "init_std": 0.05,
    }
]
"""
}
