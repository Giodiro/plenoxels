# configuration file to be used with `main.py` for video training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "legovideo_reg0.1",
    "logdir": "./logs",

    # Data settings
    #"data_downsample": 1.0,
    #"data_dirs": ["/data/DATASETS/VidNerf/lego_video"],
    "data_downsample": 8,
    "data_dirs": ["/data/DATASETS/VidNerf/coffee_martini"],
    # Data settings for 360
    "max_train_cameras": 20,
    "max_test_cameras": 1,
    "max_train_tsteps": 10,
    "max_test_tsteps": None,
    # Data settings for LLFF
    "subsample_time": 0.1,

    # Optimization settings
    "num_epochs": 5,
    "batch_size": 4096,
    "scheduler_type": "cosine",
    "optim_type": "adam",
    "lr": 8e-3,

    "regnerf_weight_start": 0.0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 700,

    "plane_tv_weight": 0.0,
    "plane_tv_what": "Gcoords",

    "l1density_weight": 0.000,

    "volume_tv_weight": 0.00,
    "volume_tv_npts": 100,
    "volume_tv_patch_size": 8,
    "volume_tv_what": "Gcoords",

    # Training settings
    "train_fp16": True,
    "save_every": 1,
    "valid_every": 1,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 2,
    "n_intersections": 400,
    "spacing_fn": "linear",
    "single_jitter": True,

    # Model settings
    "sh": True,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 10,
        "time_reso": 10,
        "time_rank": 20,
        "init_std": 0.2,
    },
    {
        "input_coordinate_dim": 4,
        "resolution": [6, 6, 6, 6],
        "feature_dim": 28,
        "init_std": 0.001
    }
]
"""
}
