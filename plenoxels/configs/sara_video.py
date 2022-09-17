# configuration file to be used with `main.py` for video training
config = {
    # "expname": "legovideo20views_regtaudepthweightedacc_1_3framesrank3",
    "expname": "testrelu_reg10",
    "logdir": "./logs",

    # Data settings
    "data_downsample": 3.0,
    "data_dirs": ["/home/sfk/data/3DVideo/lego_video"],
    # "data_dir": "/data/datasets/3DVideo/coffee_martini",
    # "data_dir": "/data/datasets/nerf/data/nerf_synthetic/lego",

    # Data settings for 360
    "max_train_cameras": 20,
    "max_test_cameras": 1,
    "max_train_tsteps": 2,
    "max_test_tsteps": 2,
    # Data settings for LLFF
    "subsample_time": 0.1,


    # Optimization settings
    "num_epochs": 5,
    "regnerf_weight": 10.0,
    "scheduler_type": None,
    "batch_size": 3500,  
    "optim_type": "adam",
    "lr": 2e-3,
    
    # Training settings
    "train_fp16": True,
    "save_every": 1,
    "valid_every": 1,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 2,
    "n_intersections": 128,
    "spacing_fn": "linear",
    "single_jitter": True,

    # Model settings
    # Giac has G init_std=0.01
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 20,
        "time_reso": 3,
        "time_rank": 3,
        "init_std": 0.1,  
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
