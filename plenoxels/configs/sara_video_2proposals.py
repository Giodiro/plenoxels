# configuration file to be used with `main.py` for video training
config = {
    # "expname": "downsample2_multiscale80.64.64to256.150_rank2_triplane256.96.48_reg60k_isg60k_ist60k_lr0.01warmupstepmany_timesmooth1_planetv0.001_anneal1k_bs4096",
    # "expname": "downsample4_F5_6_multiscale80.64.64to256.150_rank2_triplane256.96.48_reg60k_isg30k_ist30k_lr0.02warmupstepmany_timesmooth0_planetv0_anneal1k_bs4096",
    # "expname": "testproposal",
    "expname": "noF_nn_unregularized_rank2_downsample4_reso64to512",
    "logdir": "./logs/salmonvideo",

    # Data settings
    "data_downsample": 4,
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
    "isg_step": 60001,
    "ist_step": 90001,

    # Optimization settings
    "num_steps": 120001,
    # "num_steps": 2001,
    # "l1_appearance_planes_reg" : 0.1,
    "floater_loss": 0.0000,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0.000,  
    "time_smoothness_weight": 0.000,
    "l1_plane_color_reg": 0,
    "l1_plane_density_reg": 0.0,  # Try this
    "l1density_weight": 0,  # Not used for video yet
    "volume_tv_weight": 0.0,  # Not used for video yet
    "volume_tv_npts": 1024,  # Not used for video yet
    "volume_tv_what": "Gcoords",  # Not used for video yet
    "scheduler_type": "warmup_step_many", 
    "batch_size": 4096,  
    "optim_type": "adam",
    "lr": 0.01,
    
    # Training settings
    "train_fp16": True,
    "save_every": 60000,
    "valid_every": 60000,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "fixed",
    "num_sample_multiplier": 2,
    "n_intersections": 48,
    "spacing_fn": "linear",  # reciprocal. Seems to not be used if proposal sampling is used
    "single_jitter": False,
    # Proposal sampling settings
    "histogram_loss_weight": 1,  # this should be set > 0 when using proposal sampling
    "density_field_resolution": [128, 256],
    "density_field_rank": 10,
    "num_proposal_samples": [256, 96],
    "density_activation": "trunc_exp",  # can be 'relu' or 'trunc_exp'
    "density_model": "triplane",  # Can be triplane or hexplane

    # Model settings
    "sh": False,
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
        "resolution": [80, 64, 64, 150],
        "rank": 2,
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
