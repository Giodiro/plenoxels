# configuration file to be used with `main.py` for video training
config = {

    #"expname": "hexplane_lr001_tv0_histloss01_proposal128x256_256x96_ninsect48",
    #"expname": "hexplane_lr001_tv0_histloss001_ninsect128",
    #"expname": "hexplane_no_testtime_optim_l1_appearance_planes_reg0.1_add_appearance",
    #"expname": "debug_rank2",
    "expname" : "hexplane_lr0005",
    "logdir": "./logs/trevi/nov12",

    # Data settings
    "data_downsample": 1.0,
    "data_dirs": ["/home/warburg/data/phototourism/trevi"],

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
    "num_steps": 120_001,
    "floater_loss": 0.0000,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0.0,  
    "l1density_weight": 0,  # Not used for video yet
    "volume_tv_weight": 0.0,  # Not used for video yet
    "volume_tv_npts": 1024,  # Not used for video yet
    "volume_tv_what": "Gcoords",  # Not used for video yet
    "scheduler_type": "cosine",
    "batch_size": 4096,  
    "optim_type": "adam",
    "lr": 0.005,
    "use_F": False,
    # proposal sampling
    "histogram_loss_weight": 0,  # this should be set > 0 when using proposal sampling
    "density_field_resolution": [128, 256],
    "density_field_rank": 10,
    "num_proposal_samples": [256, 96],
    "density_activation": "trunc_exp",
    "density_model": "triplane",  # Can be triplane or hexplane
    "l1_appearance_planes_reg" : 1,
    
    # Training settings
    "train_fp16": True,
    "save_every":  10000,
    "valid_every": 10000,
    "save_video": True,
    "add_rank_steps": [],
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "fixed",
    "num_sample_multiplier": 2,
    "n_intersections": 440,
    "spacing_fn": "linear",
    "single_jitter": False,
    

    # Model settings
    "sh": False,
    "appearance_code_size": 32,
    "color_net": 2,
    "upsample_time_resolution": [],
    # "upsample_time_steps": [6000],  # DyNerf does 300K iterations with keyframes, with lr 5e-4
    "upsample_time_steps": [],
    #"upsample_resolution": [],
    "multiscale_res": [1, 2, 4, 8],
    "grid_config": """
[
    {
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 64,
        "grid_dimensions": 2,
        "resolution": [80, 40, 20, 1708], 
        "rank": 1,
        "time_reso": 1708,
    },
    {
        "input_coordinate_dim": 2,
        "resolution": [6, 6, 6, 6, 6],
        "feature_dim": 64,
        "init_std": 0.001
    }
]
"""
}
# trevi : 1708
# sacre : 1200
