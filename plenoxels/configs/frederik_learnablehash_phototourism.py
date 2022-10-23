# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "phototourism_contrast_bds",
    "logdir": "./logs",

    # Data settings
    "data_resolution": None,
    "data_downsample": 3,
    "data_dirs": ["/work3/frwa/data/phototourism/sacre"],  # CHANGE ME!
    # Data settings for 360
    "max_tr_frames": None,
    "max_ts_frames": 10,
    # Data settings for LLFF
    "hold_every": 8,
    "keyframes": False,
    "ist_step": -1,

    # Optimization settings
    "num_steps": 40001,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0.00,  
    "l1density_weight": 0,  # Not used for video yet
    "volume_tv_weight": 0.0,  # Not used for video yet
    "volume_tv_npts": 1024,  # Not used for video yet
    "volume_tv_what": "Gcoords",  # Not used for video yet
    "scheduler_type": "step",
    "batch_size": 4096,  
    "optim_type": "adam",
    "lr": 0.1,

    # Training settings
    "train_fp16": True,
    "save_every": 5000,
    "valid_every": 5000,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 2,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 400,  # Used when raymarch_type is 'fixed'
    "spacing_fn": "log",
    "single_jitter": True,

    # Model settings
    "sh": True,
    "density_threshold": 1e-4,
    "dmask_update": [],
    "upsample_steps": [],
    "upsample_time_resolution": [],
    "upsample_time_steps": [],
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
        "time_reso": 1179,
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
