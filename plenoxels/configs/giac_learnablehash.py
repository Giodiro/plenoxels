# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "lego_test",
    "logdir": "./logs",

    # Data settings
    "data_resolution": None,
    "data_downsample": 1,
    "data_dirs": ["/data/DATASETS/SyntheticNerf/lego"],
    #"data_dirs": ["/data/DATASETS/LLFF/fern"],
    # Data settings for 360
    "max_tr_frames": 100,
    "max_ts_frames": 10,
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_steps": 20000,
    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 1e-2,

    # Regularization
    "floater_loss": 0,
    "plane_tv_weight": 0.000,
    "plane_tv_what": "Gcoords",
    "l1density_weight": 0.000,
    "volume_tv_weight": 0.00,
    "volume_tv_npts": 100,
    "volume_tv_patch_size": 8,
    "volume_tv_what": "Gcoords",
    "l1_plane_color_weight": 0.0,
    "l1_plane_density_weight": 0.000,

    # Training settings
    "train_fp16": True,
    "save_every": 30_000,
    "valid_every": 3150,
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
    "sh_decoder_type": "manual",  # can be 'tcnn' or 'manual'
    "density_threshold": 4e-4,
    "dmask_update": [],
    "upsample_steps": [],
    #"dmask_update": [2000, 4000],
    #"upsample_steps": [2000, 3000, 4000, 5500, 7000],
    "upsample_resolution": [],
    "density_multiplier": 1,
    "use_F": False,
    "density_activation": "trunc_exp",  # can be 'relu' or 'trunc_exp'

    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 28,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 2,
    },
]
"""
}
