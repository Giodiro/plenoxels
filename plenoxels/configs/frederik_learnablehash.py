# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "vis_planes",
    "logdir": "./logs/lego",

    # Data settings
    "data_resolution": None,
    "data_downsample": 4,
    "data_dirs": ["/work3/frwa/lego"],  # CHANGE ME!
    # Data settings for 360
    "max_tr_frames": None,
    "max_ts_frames": 10,
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_steps": 15001,
    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": None,
    "optim_type": "adam",
    "lr": 0.01,

    # Regularization settings
    "regnerf_weight_start": 0.0,
    "regnerf_weight_end": 0.0,
    "floater_loss" : 0.0,
    "regnerf_weight_max_step": 700,

    "plane_tv_weight": 0.00,
    "plane_tv_what": "Gcoords",

    "l1density_weight": 0.000,

    "volume_tv_weight": 0.00,
    "volume_tv_npts": 100,
    "volume_tv_patch_size": 8,
    "volume_tv_what": "Gcoords",

    # Training settings
    "train_fp16": True,
    "save_every": 1000,
    "valid_every": 1000,
    "save_outputs": True,
    "transfer_learning": False,

    # Raymarching settings
    "raymarch_type": "fixed",
    "num_sample_multiplier": 2,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 400,  # Used when raymarch_type is 'fixed'
    "spacing_fn": "reciprocal",
    "single_jitter": True,

    # Model settings
    "sh": True,
    "density_threshold": 1e-4,
    "dmask_update": [],
    "upsample_steps": [],
    "upsample_resolution": [],
    "upsample_F_steps": [],
    "density_multiplier": 1,
    "use_F": False,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 28,
        "grid_dimensions": 2,
        "resolution": [512, 512, 512], 
        "rank": 3,
    },
    {
        "input_coordinate_dim": 5,
        "resolution": [6, 6, 6, 6, 6],
        "feature_dim": 28,
        "init_std": 0.001
    }
]
"""
}
