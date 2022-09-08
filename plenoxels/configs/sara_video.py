# configuration file to be used with `run_video.py`
samples_per_voxel = 2
resolution = 128
config = {
    "expname": "test",
    "data_downsample": 4.0,
    "data_dir": "/data/datasets/3DVideo/coffee_martini",
    # "data_dir": "/data/datasets/nerf/data/nerf_synthetic/lego",

    "subsample_time_train": 0.2,
    "batch_size": 10000,  
    "num_batches_per_dset": 1,
    "num_epochs": 1,
    "scheduler_type": None,
    "optim_type": "adam",
    "model_type": "learnable_hash",
    "logdir": "./logs",
    "train_fp16": True,
    "save_every": 1,
    "valid_every": 1,
    "transfer_learning": False,

    "lr": 2e-3,

    "raymarch_type": "voxel_size",
    "sampling_resolution": resolution,
    "num_sample_multiplier": samples_per_voxel,
    "n_intersections": resolution,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 4,
        "grid_dimensions": 2,
        "resolution": 128,
        "rank": 10,
        "time_reso": 2,
        "time_rank": 1,
        "init_std": 0.1,
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
