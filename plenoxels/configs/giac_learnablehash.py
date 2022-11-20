# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "lego_test_nerfacc_noF",
    "logdir": "./logs",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 1,
    "data_dirs": ["/data/DATASETS/SyntheticNerf/lego"],
    #"data_dirs": ["/data/DATASETS/LLFF/fern"],
    # Data settings for 360
    "max_tr_frames": None,
    "max_ts_frames": 10,
    # Data settings for 360
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_steps": 35000,
    "scheduler_type": "warmup_cosine",
    "lr": 1e-2,
    "cone_angle": 0.00,
    "optim_type": "adam",

    "alpha_threshold": 1e-3,

    "plane_tv_weight": 0.000,
    "plane_tv_what": "Gcoords",

    "l1density_weight": 0.000,

    "volume_tv_weight": 0.000,
    "volume_tv_npts": 100,
    "volume_tv_patch_size": 8,
    "volume_tv_what": "Gcoords",

    # Training settings
    "train_fp16": True,
    "save_every": 35000,
    "valid_every": 5000,
    "save_outputs": False,
    "transfer_learning": False,

    # Raymarching settings
    "sample_batch_size": 1 << 20,
    "n_samples": 1024,

    # Model settings
    "sh": False,
    "use_F": False,
    "density_activation": "trunc_exp",
    "density_threshold": 1e-2,
    "multiscale_res": [1, 2],

    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 32,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 2,
    },
]
"""
}
