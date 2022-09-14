# configuration file to be used with `run_video.py`
config = {
    "expname": "legovideo_reg0.1",
    # Data settings
    "data_downsample": 2.0,
    "data_dirs": ["/home/sfk/data/3DVideo/lego_video"],
    # Data settings for 360
    "max_train_cameras": 50,
    "max_test_cameras": 1,
    "max_train_tsteps": 10,
    "max_test_tsteps": None,
    # Data settings for LLFF
    "subsample_time": 0.1,

    # Optimization settings
    "num_epochs": 1,
    "batch_size": 4096,
    "regnerf_weight": 0.1,
    "scheduler_type": None,
    "optim_type": "adam",
    "lr": 2e-3,

    # Training settings
    "train_fp16": True,
    "save_every": 1,
    "valid_every": 1,
    "save_video": True,
    "save_outputs": True,

    "logdir": "./logs",

    "model_type": "learnable_hash",
    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 2,
    "n_intersections": 400,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 3,
        "resolution": [128, 128, 128],
        "rank": 1,
        "time_reso": 10,
        "time_rank": 5,
        "init_std": 0.01,
    },
    {
        "input_coordinate_dim": 4,
        "resolution": [8, 8, 8, 8],
        "feature_dim": 32,
        "init_std": 0.05
    }
]
"""
}
