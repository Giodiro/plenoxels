# configuration file to be used with `main.py` for video training
config = {
    # "expname": "newmodel500.500.500.150_noF_rank2_reg30k_isg30k_ist30k_lr0.01cosine_l1sigma0.1_planetv0.05_sample400_init0.1to0.5",
    "expname": "multiscale64to512.150_rank4_proposal256.128_reg20k_isg20k_ist20k_lr0.01_timesmooth1_planetv0.001_scale0.6xyshift2z_normalizeraysd_annealproposalweightsz",
    # "expname": "testproposal",
    "logdir": "./logs/salmonvideo",

    # Data settings
    "data_downsample": 4.0,
    # "data_dirs": ["/home/sfk/data/3DVideo/lego_video"],
    # "data_dirs": ["/home/sfk/data/3DVideo/coffee_martini"],
    "data_dirs": ["/home/sfk/data/3DVideo/flame_salmon_1"],

    # Data settings for 360
    "max_train_cameras": 20,
    "max_test_cameras": 1,
    "max_train_tsteps": 2,
    "max_test_tsteps": 2,
    # Data settings for LLFF
    "keyframes": False,
    "isg_step": 20000,
    "ist_step": 40000,

    # Optimization settings
    "num_steps": 60001,
    # "num_steps": 2001,
    "floater_loss": 0.0000,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0.01,  
    "time_smoothness_weight": 1.0,
    "l1_plane_color_reg": 0,
    "l1_plane_density_reg": 0,  # Try this
    "l1density_weight": 0,  # Not used for video yet
    "volume_tv_weight": 0.0,  # Not used for video yet
    "volume_tv_npts": 1024,  # Not used for video yet
    "volume_tv_what": "Gcoords",  # Not used for video yet
    "scheduler_type": "cosine", # "step"
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
    "n_intersections": 128,
    "spacing_fn": "linear",  # reciprocal. Seems to not be used if proposal sampling is used
    "single_jitter": False,
    # Proposal sampling settings
    "histogram_loss_weight": 1,  # this should be set > 0 when using proposal sampling
    "density_field_resolution": [128],
    "density_field_rank": 10,
    "num_proposal_samples": [256],

    # Model settings
    "sh": True,
    "upsample_time_resolution": [150],
    # "upsample_time_steps": [6000],  # DyNerf does 300K iterations with keyframes, with lr 5e-4
    "upsample_time_steps": [-1],
    "use_trainable_rank": False,
    "add_rank_steps": [-1],
    "use_F": False,
    "multiscale_res": [1, 2, 4, 8],
    "grid_config": """
[
    {
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 28,
        "grid_dimensions": 2,
        "resolution": [64, 64, 64, 150],
        "rank": 1,
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
