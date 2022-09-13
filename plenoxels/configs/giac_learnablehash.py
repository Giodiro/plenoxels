# configuration file to be used with `run_multi_scene.py`
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "lego_test",
    "data_resolution": None,
    "data_downsample": 1.5625,
    "data_dirs": ["/data/DATASETS/SyntheticNerf/lego"],#["/data/DATASETS/LLFF/fern"],

    "max_tr_frames": None,
    "max_ts_frames": 10,
    "hold_every": 8,

    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "num_epochs": 10,
    "dmask_update": [4000, 12000],
    "density_threshold": 1e-4,

    "scheduler_type": None,
    "optim_type": "adam",
    "lr": 2e-3,

    "logdir": "./logs",
    "train_fp16": True,
    "save_every": 10,
    "valid_every": 1,
    "save_outputs": True,
    "transfer_learning": False,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "sampling_resolution": 128,
    "num_sample_multiplier": 2,
    "n_intersections": 400,
    "spacing_fn": "linear",
    "single_jitter": True,

    "model_type": "learnable_hash",
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 3,
        "resolution": [128, 128, 128],
        "rank": 1,
        "init_std": 0.01,
    },
    {
        "input_coordinate_dim": 4,
        "resolution": [10, 10, 10, 10],
        "feature_dim": 32,
        "init_std": 0.05
    }
]
"""
}
