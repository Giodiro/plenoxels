# configuration file to be used with `main.py` for video training
config = {
    # "expname": "legovideo20views_regdepthweightedacc_400_0.1_512_3framesreso3rank20",
    "expname": "test_keyframes4k_isg10k_ist20k",
    # "expname": "testspeed",
    # "expname": "testrelu_sameranktimereso128_llff",
    "logdir": "./logs/coffeevideo",

    # Data settings
    "data_downsample": 8.0,
    # "data_dirs": ["/home/sfk/data/3DVideo/lego_video"],
    "data_dirs": ["/home/sfk/data/3DVideo/coffee_martini"],
    # "data_dir": "/data/datasets/nerf/data/nerf_synthetic/lego",

    # Data settings for 360
    "max_train_cameras": 20,
    "max_test_cameras": 1,
    "max_train_tsteps": 2,
    "max_test_tsteps": 2,
    # Data settings for LLFF
    "keyframes": True,
    "isg": True,
    "ist_step": 10000,

    # Optimization settings
    "num_steps": 20000,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0,  # Not used for video yet
    "l1density_weight": 0,  # Not used for video yet
    "volume_tv_weight": 0.0,  # Not used for video yet
    "volume_tv_npts": 1024,  # Not used for video yet
    "volume_tv_what": "Gcoords",  # Not used for video yet
    "scheduler_type": None,
    "batch_size": 4096,  
    "optim_type": "adam",
    "lr": 1e-2,
    
    # Training settings
    "train_fp16": True,
    "save_every": 5000,
    "valid_every": 2000,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 2,
    "n_intersections": 128,
    "spacing_fn": "linear",
    "single_jitter": True,

    # Model settings
    "sh": True,
    "upsample_time_resolution": [150],
    # "upsample_time_steps": [3649], # 2 epochs with keyframes # DyNerf does 300K iterations with keyframes, with lr 5e-4
    "upsample_time_steps": [4000],
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 5,
        "grid_dimensions": 2,
        "resolution": [200, 200, 128],
        "rank": 30,
        "time_reso": 30,
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
