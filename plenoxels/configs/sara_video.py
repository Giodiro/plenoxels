# configuration file to be used with `main.py` for video training
config = {
    # "expname": "legovideo20views_regdepthweightedacc_400_0.1_512_3framesreso3rank20",
    "expname": "test",
    # "expname": "testrelu_sameranktimereso128_llff",
    "logdir": "./logs/legovideo",

    # Data settings
    "data_downsample": 3.0,
    "data_dirs": ["/home/sfk/data/3DVideo/lego_video"],
    # "data_dirs": ["/home/sfk/data/3DVideo/coffee_martini"],
    # "data_dir": "/data/datasets/nerf/data/nerf_synthetic/lego",

    # Data settings for 360
    "max_train_cameras": 20,
    "max_test_cameras": 1,
    "max_train_tsteps": 2,
    "max_test_tsteps": 2,
    # Data settings for LLFF
    "subsample_time": 0.1,


    # Optimization settings
    "num_epochs": 10,
    "regnerf_weight_start": 10,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0,  # Not used for video yet
    "l1density_weight": 0,  # Not used for video yet
    "volume_tv_weight": 0.0,  # Not used for video yet
    "volume_tv_npts": 1024,  # Not used for video yet
    "scheduler_type": None,
    "batch_size": 4096,  
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
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 20,
        "time_reso": 3,
        "init_std": 0.2,  
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
