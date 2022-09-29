# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
import numpy as np
config = {
    "expname": "lego_test0voltv",
    "logdir": "./logs",

    # Data settings
    "data_resolution": None,
    "data_downsample": 3,
    "data_dirs": ["/home/sfk/data/nerf_synthetic/lego"],
    # "data_dirs": ["/home/sfk/data/nerf_llff_data/fern"],
    # Data settings for 360
    "max_tr_frames": 20,
    "max_ts_frames": 10,
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_epochs": 10,
    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": None,
    "optim_type": "adam",
    "lr": 2e-3,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0,
    "regnerf_weight_max_step": 512,
    "l1density_weight": 0,
    "plane_tv_weight": 0,
    "volume_tv_weight": 0.0,
    "volume_tv_npts": 1024,

    # Training settings
    "train_fp16": True,
    "save_every": 10,
    "valid_every": 1,
    "save_outputs": True,
    "transfer_learning": False,

    # Raymarching settings
    "raymarch_type": "voxel_size",
    "num_sample_multiplier": 2,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 440,  # Used when raymarch_type is 'fixed'
    "spacing_fn": "linear",
    "single_jitter": True,

    # Model settings
    "density_threshold": 1e-4,
    "dmask_update": [np.inf],
    "density_multiplier": 1,
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 5,
        "grid_dimensions": 2,
        "resolution": [128, 128, 128],
        "rank": 10,
        "init_std": 0.2,  
    },
    {
        "input_coordinate_dim": 5,
        "resolution": [6, 6, 6, 6, 6],
        "feature_dim": 32,
        "init_std": 0.05
    }
]
"""
}












# # configuration file to be used with `run_multi_scene.py`
# # the configuration must be specified in a dictionary called `config`.
# import numpy as np

# samples_per_voxel = 2
# resolution = 256
# config = {
#     "expname": "test",
#     "data_resolution": None,
#     "data_downsample": 1.562,
#     "data_dirs": ["/data/datasets/nerf/data/nerf_synthetic/lego"],
#     # "data_dirs": ["/data/datasets/nerf/data/nerf_synthetic/lego", 
#     #             "/data/datasets/nerf/data/nerf_synthetic/materials",
#     #             "/data/datasets/nerf/data/nerf_synthetic/mic",
#     #             "/data/datasets/nerf/data/nerf_synthetic/ship"],

#     "max_tr_frames": None,
#     "max_ts_frames": 10,
#     "batch_size": 2000,
#     "num_batches_per_dset": 1,
#     "num_epochs": 10,
#     "scheduler_type": None,
#     "optim_type": "adam",
#     "model_type": "learnable_hash",
#     "logdir": "./logs",
#     "train_fp16": True,
#     "save_every": 1,
#     "valid_every": 1,
#     "transfer_learning": False,

#     "lr": 2e-3,

#     "raymarch_type": "voxel_size",
#     "sampling_resolution": resolution,
#     "num_sample_multiplier": samples_per_voxel,
#     "n_intersections": resolution,
#     "grid_config": """
# [
#     {
#         "input_coordinate_dim": 3,
#         "output_coordinate_dim": 4,
#         "grid_dimensions": 2,
#         "resolution": 256,
#         "rank": 20,
#         "init_std": 0.1,
#     },
#     {
#         "input_coordinate_dim": 4,
#         "resolution": 8,
#         "feature_dim": 32,
#         "init_std": 0.05
#     }
# ]
# """
# }
