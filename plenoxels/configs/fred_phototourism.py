# configuration file to be used with `main.py` for video training
config = {

    "expname": "optimizer_appearance_code",
    "logdir": "./logs/trevi",

    # Data settings
    "data_downsample": 1.0,
    "data_dirs": ["/work3/frwa/data/phototourism/trevi"],

    # Data settings for 360
    #"max_train_cameras": 20,
    #"max_test_cameras": 1,
    #"max_train_tsteps": 2,
    #"max_test_tsteps": 2,
    # Data settings for LLFF
    "keyframes": False,
    "isg": False,
    "ist_step": -1,
    "isg_step": -1,

    # Optimization settings
    # "num_steps": 40001,
    "num_steps": 60001,
    "floater_loss": 0.0000,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0.000,  
    "l1density_weight": 0,  # Not used for video yet
    "volume_tv_weight": 0.0,  # Not used for video yet
    "volume_tv_npts": 1024,  # Not used for video yet
    "volume_tv_what": "Gcoords",  # Not used for video yet
    "scheduler_type": "cosine",
    "batch_size": 4096,  
    "optim_type": "adam",
    "lr": 0.01,
    "use_F": False,
    
    # Training settings
    "train_fp16": False,
    "save_every": 5000,
    "valid_every": 5000,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "fixed",
    "num_sample_multiplier": 2,
    "n_intersections": 800,
    "spacing_fn": "linear",
    "single_jitter": False,
    

    # Model settings
    "sh": True,
    "upsample_time_resolution": [],
    # "upsample_time_steps": [6000],  # DyNerf does 300K iterations with keyframes, with lr 5e-4
    "upsample_time_steps": [],
    #"upsample_resolution": [],
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 28,
        "grid_dimensions": 2,
        "resolution": [256, 256, 256],
        "rank": 2,
        "time_reso": 3191,
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
# trevi : 1169