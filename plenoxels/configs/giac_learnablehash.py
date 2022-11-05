# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "lego_test_propsampling",
    "logdir": "./logs",

    # Data settings
    "data_resolution": None,
    "data_downsample": 1,
    "data_dirs": ["/data/DATASETS/SyntheticNerf/lego"],
    #"data_dirs": ["/data/DATASETS/LLFF/fern"],
    # Data settings for 360
    "max_tr_frames": 100,
    "max_ts_frames": 10,
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_steps": 30_001,
    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": "warmup_step_many",
    "optim_type": "adam",
    "lr": 2e-2,

    # Regularization
    "floater_loss": 0,
    "plane_tv_weight": 0.004,
    "plane_tv_what": "Gcoords",
    "l1density_weight": 0.000,
    "volume_tv_weight": 0.00,
    "volume_tv_npts": 100,
    "volume_tv_patch_size": 8,
    "volume_tv_what": "Gcoords",
    "l1_plane_color_weight": 0.0,
    "l1_plane_density_weight": 0.000,
    "histogram_loss_weight": 1.0,  # this should be set > 0 when using proposal sampling

    # Training settings
    "train_fp16": True,
    "save_every": 60000,
    "valid_every": 10000,
    "save_outputs": True,
    "transfer_learning": False,

    # Raymarching settings
    "raymarch_type": "fixed",
    "num_sample_multiplier": 2,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 64,  # Used when raymarch_type is 'fixed'
    "spacing_fn": "linear",
    "single_jitter": False,
    # proposal sampling
    "density_field_resolution": [128, 128, 128],
    "density_field_rank": 10,
    "num_proposal_samples": 128,

    # Model settings
    "sh": True,
    "sh_decoder_type": "manual",  # can be 'tcnn' or 'manual'
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
        "output_coordinate_dim": 28,
        "grid_dimensions": 2,
        "resolution": [64, 64, 64],#[256, 192, 168],
        "rank": 1,
    },
    {
        "input_coordinate_dim": 5,
        "resolution": [6, 6, 6, 6, 6],
        "feature_dim": 28,
        "init_std": 0.001,
    }
]
"""
}
