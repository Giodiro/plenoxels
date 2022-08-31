config = {
    "expname": "test",
    "data_resolution": None,
    "data_downsample": 1.562,
    "data_dirs": ["/data/DATASETS/SyntheticNerf/lego"],
    "max_tr_frames": None,
    "max_ts_frames": 10,
    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "num_epoch": 10,
    "scheduler_type": None,
    "optim_type": "adam",
    "model_type": "learnable_hash",
    "logdir": "./logs",
    "train_fp16": True,
    "save_every": 10,
    "valid_every": 1,
    "transfer_learning": False,

    "lr": 2e-3,

    "n_intersections": 256,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 3,
        "resolution": 128,
        "rank": 1,
        "init_std": 0.01,
    },
    {
        "input_coordinate_dim": 4,
        "resolution": 8,
        "feature_dim": 32,
        "init_std": 0.05
    }
]
"""
}
