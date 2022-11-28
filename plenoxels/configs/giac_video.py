# configuration file to be used with `main.py` for video training
config = {
    "expname": "test_nerfacc",
    "logdir": "./logs/flame_salmon",
    "device": "cuda:0",
    "wandb": False,

    # Data settings
    "data_downsample": 2.0,
    "data_dirs": ["/data/DATASETS/VidNerf/flame_salmon"],
    # "data_dirs": ["/data/DATASETS/VidNerf/lego_video"],

    # Data settings for 360
    "max_train_cameras": 100,
    "max_test_cameras": 1,
    "max_train_tsteps": 5,  # determines time-downsampling for keyframes
    "max_test_tsteps": None,
    # Data settings for LLFF
    "keyframes": False,
    "isg": False,
    "isg_step": -1,
    "ist_step": 90000,

    # Optimization settings
    "num_steps": 50_000,
    "scheduler_type": "warmup_cosine",
    "lr": 1e-2,
    "cone_angle": 0.00,
    "optim_type": "adam",

    # Regularization
    "plane_tv_weight": 0.000,
    "l1density_weight": 0,
    "volume_tv_weight": 0,
    "volume_tv_npts": 0,

    # Training settings
    "train_fp16": True,
    "save_every": 5_000,
    "valid_every": 5_000,
    "save_outputs": True,

    # Raymarching settings
    "sample_batch_size": 1 << 18,
    "n_samples": 1024,
    "alpha_threshold": 0.0,
    "density_threshold": 1e-2,
    "early_stop_eps": 1e-3,

    # Model settings
    "sh": False,
    "use_F": False,
    "density_activation": "trunc_exp",
    "multiscale_res": [1, 2, 4],
    "concat_features": True,
    "occupancy_grid_resolution": [128, 128, 128],
    "grid_config": """
[
    {
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 16,
        "grid_dimensions": 2,
        "resolution": [64, 64, 64, 150],
    }
]
"""
}
