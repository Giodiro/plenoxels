# configuration file to be used with `main.py` for video training
config = {
    "expname": "salmon_test",
    "logdir": "./logs",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 1.0,
    #"data_dirs": ["/data/DATASETS/VidNerf/flame_salmon"],
    "data_dirs": ["/data/DATASETS/VidNerf/lego_video"],

    # Data settings for 360
    "max_train_cameras": 100,
    "max_test_cameras": 1,
    "max_train_tsteps": 5,  # determines time-downsampling for keyframes
    "max_test_tsteps": None,
    # Data settings for LLFF
    "keyframes": True,
    "isg": True,
    "ist_step": 150000,

    # Optimization settings
    "num_steps": 30000,
    "plane_tv_weight": 0,  # Not used for video yet
    "scheduler_type": None,
    "batch_size": 4096,
    "optim_type": "adam",
    "lr": 1e-2,

    # Regularization
    "plane_tv_weight": 0.000,
    # Regularization - unused
    "l1density_weight": 0,
    "volume_tv_weight": 0,
    "volume_tv_npts": 0,

    # Training settings
    "train_fp16": True,
    "save_every": 500000,
    "valid_every": 5000,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "sample_batch_size": 1 << 18,
    "n_samples": 1024,
    "alpha_threshold": 1e-3,
    "cone_angle": 0.0,

    # Model settings
    "sh": True,
    "density_threshold": 1e-2,
    "upsample_time_resolution": [12],
    "upsample_time_steps": [10000],  # DyNerf does 300K iterations with keyframes, with lr 5e-4
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 10,
        "time_reso": 5,
    },
    {
        "input_coordinate_dim": 4,
        "resolution": [6, 6, 6, 6],
        "feature_dim": 28,
        "init_std": 0.1
    }
]
"""
}
