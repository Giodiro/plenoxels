# configuration file to be used with `main.py` for video training
config = {
    "expname": "mlp_downsample2_istonly",
    "logdir": "./logs/salmonvideo",

    # Data settings
    "data_downsample": 2.0,
    #"data_dirs": ["/data/DATASETS/VidNerf/lego_video"],
    # "data_dirs": ["/home/sfk/data/3DVideo/coffee_martini"],
    "data_dirs": ["/data/DATASETS/VidNerf/flame_salmon"],

    # Data settings for 360
    "max_train_cameras": 25,
    "max_test_cameras": 1,
    "max_train_tsteps": None,
    "max_test_tsteps": None,
    # Data settings for LLFF
    "keyframes": False,
    "isg_step": -1,
    "isg": False,
    "ist_step": 80000,

    # Optimization settings
    "num_steps": 120001,
    "floater_loss": 0.0000,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "plane_tv_weight": 0.1,
    "time_smoothness_weight": 0.1,
    "l1_plane_color_reg": 0,
    "l1_plane_density_reg": 0,
    "l1density_weight": 0,     # Not used for video yet
    "volume_tv_weight": 0.0,   # Not used for video yet
    "volume_tv_npts": 1024,    # Not used for video yet
    "volume_tv_what": "Gcoords",  # Not used for video yet
    "scheduler_type": "warmup_cosine", # "step"
    "batch_size": 4096,
    "optim_type": "adam",
    "lr": 0.01,

    # Training settings
    "train_fp16": True,
    "save_every": 120000,
    "valid_every": 30000,
    "save_video": True,
    "save_outputs": True,

    # Raymarching settings
    "raymarch_type": "fixed",
    #"num_sample_multiplier": 2,
    "n_intersections": 48,
    #"spacing_fn": "linear",  # reciprocal. Seems to not be used if proposal sampling is used
    "single_jitter": False,
    # Proposal sampling settings
    "histogram_loss_weight": 1.0,  # this should be set > 0 when using proposal sampling
    "density_model": "triplane",
    "density_field_resolution": [128, 256],
    "density_field_rank": 1,
    "num_proposal_samples": [256, 96],
    "proposal_feature_dim": 10,
    "proposal_decoder_type": "nn",  # can be 'sh' or 'nn'

    # Model settings
    "sh": False,
    "sh_decoder_type": "manual",
    "density_activation": "trunc_exp",
    "upsample_time_resolution": [150],
    # "upsample_time_steps": [6000],  # DyNerf does 300K iterations with keyframes, with lr 5e-4
    "upsample_time_steps": [-1],
    "use_trainable_rank": False,
    "add_rank_steps": [-1],
    "use_F": False,
    "multiscale_res": [1, 2, 4],
    "grid_config": """
[
    {
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 32,
        "grid_dimensions": 2,
        "resolution": [80, 64, 64, 150],
        "rank": 2,
    },
]
"""
}
