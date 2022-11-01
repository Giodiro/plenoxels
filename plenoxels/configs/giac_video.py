# configuration file to be used with `main.py` for video training
config = {
    "expname": "test_salmon_rank4",
    "logdir": "./logs",

    # Data settings
    "data_downsample": 4.0,
    # "data_dirs": ["/home/sfk/data/3DVideo/lego_video"],
    # "data_dirs": ["/home/sfk/data/3DVideo/coffee_martini"],
    "data_dirs": ["/data/DATASETS/VidNerf/flame_salmon"],

    # Data settings for 360
    "max_train_cameras": 20,
    "max_test_cameras": 1,
    "max_train_tsteps": 2,
    "max_test_tsteps": 2,
    # Data settings for LLFF
    "keyframes": False,
    "isg_step": 20000,
    "ist_step": -20000,

    # Optimization settings
    # "num_steps": 40001,
    "num_steps": 40001,
    "floater_loss": 0.0000,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0.02,
    "l1density_weight": 0,  # Not used for video yet
    "volume_tv_weight": 0.0,  # Not used for video yet
    "volume_tv_npts": 1024,  # Not used for video yet
    "volume_tv_what": "Gcoords",  # Not used for video yet
    "scheduler_type": "warmup_cosine", # "step"
    "batch_size": 4096,
    "optim_type": "adam",
    "lr": 0.01,

    # Training settings
    "train_fp16": True,
    "save_every": 10000,
    "valid_every": 10000,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "fixed",
    "num_sample_multiplier": 2,
    "n_intersections": 400,
    "spacing_fn": "reciprocal",
    "single_jitter": False,

    # Model settings
    "sh": True,
    "upsample_time_resolution": [150],
    # "upsample_time_steps": [6000],  # DyNerf does 300K iterations with keyframes, with lr 5e-4
    "upsample_time_steps": [-1],
    "use_F": False,
    "add_rank_steps": [100, 101, 102],
    "grid_config": """
[
    {
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 28,
        "grid_dimensions": 2,
        "resolution": [300, 300, 300, 150],
        "rank": 4,
    },
    {
        "input_coordinate_dim": 1,
        "resolution": [3, 3],
        "feature_dim": 28,
        "init_std": 0.001
    }
]
"""
}
