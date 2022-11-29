# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "ficus_test_wdenseweight2e-6",
    "logdir": "./logs",
    "device": "cuda:0",

    # Data settings
    "data_resolution": None,
    "data_downsample": 1,
    "data_dirs": ["/data/DATASETS/SyntheticNerf/ficus"],
    #"data_dirs": ["/data/DATASETS/LLFF/fern"],
    # Data settings for 360
    "max_tr_frames": 100,
    "max_ts_frames": 50,
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_steps": 30_001,
    "batch_size": 4096,
    "eval_batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 1e-2,

    # Regularization
    "floater_loss": 0,
    "plane_tv_weight": 2e-5,
    "plane_tv_what": "Gcoords",
    "density_plane_tv_weight": 0,
    #"l1density_weight": 0.000,
    #"volume_tv_weight": 0.00,
    #"volume_tv_npts": 100,
    "volume_tv_patch_size": 8,
    "volume_tv_what": "Gcoords",
    "l1_plane_color_weight": 0.0,
    "l1_plane_density_weight": 0.000,
    "histogram_loss_weight": 1.0,  # this should be set > 0 when using proposal sampling

    # Training settings
    "train_fp16": True,
    "save_every": 30000,
    "valid_every": 10000,
    "save_outputs": False,
    "transfer_learning": False,

    # Raymarching settings
    "raymarch_type": "fixed",
    #"num_sample_multiplier": 2,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 64,  # Used when raymarch_type is 'fixed'
    #"spacing_fn": "linear",
    "single_jitter": False,
    # proposal sampling
    "density_field_resolution": [128, 256],
    "density_field_rank": 1,
    "num_proposal_samples": [128, 96],
    "proposal_decoder_type": "nn",
    "proposal_feature_dim": 10,

    # Model settings
    "sh": False,
    #"sh_decoder_type": "manual",  # can be 'tcnn' or 'manual'
    "density_threshold": 4e-4,
    "dmask_update": [],
    "upsample_steps": [],
    "upsample_resolution": [],
    "density_multiplier": 1,
    "use_F": False,
    "density_activation": "trunc_exp",  # can be 'relu' or 'trunc_exp'
    "multiscale_res": [1, 2, 4],

    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 64,
        "grid_dimensions": 2,
        "resolution": [64, 64, 64],
        "rank": 1,
    },
]
"""
}
