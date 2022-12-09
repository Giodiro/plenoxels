# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
import numpy as np
config = {
    # "expname": "average_64.64at1k.64at2k.64at4k_nolrdecay",
    # "expname": "average_64.64.64.64_nolrdecay",
    "expname": "ship60k_linearsigmadecoderbothmainandsampler",
    # "logdir": "./logs/fullresofern",
    # "logdir": "./logs/mic_debugresolution",
    "logdir": "./logs/learnedbasis",

    # Data settings
    "data_resolution": None,
    "data_downsample": 1,
    "data_dirs": ["/home/sfk/data/nerf_synthetic/ship"],
    # "data_dirs": ["/home/sfk/data/nerf_llff_data/room"],
    # Data settings for 360
    "max_tr_frames": None,
    "max_ts_frames": 10,
    # Data settings for LLFF
    "hold_every": 8,

    # Optimization settings
    "num_steps": 60001,
    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 0.01,
    "regnerf_weight_start": 0,
    "regnerf_weight_end": 0.0,
    "regnerf_weight_max_step": 512,
    "l1density_weight": 0,
    "plane_tv_weight": 2e-5,
    "plane_tv_weight_sigma": 0.0,  # Tune this
    "plane_tv_weight_sh": 0.0,  # Tune this
    "volume_tv_weight": 0.0,
    "volume_tv_npts": 1024,
    "floater_loss": 0,

    # Training settings
    "train_fp16": True,
    "save_every": 10000,
    "valid_every": 10000,
    "save_outputs": True,
    "transfer_learning": False,

    # Raymarching settings
    "raymarch_type": "fixed",
    "num_sample_multiplier": 2,  # Used when raymarch_type is 'voxel_size'
    "n_intersections": 48,  # Used when raymarch_type is 'fixed'
    "spacing_fn": "linear",
    "single_jitter": False,
    # Proposal sampling settings
    "histogram_loss_weight": 1,  # this should be set > 0 when using proposal sampling
    "density_field_resolution": [128, 256],
    "density_field_rank": 10,
    "num_proposal_samples": [256, 96],
    "density_activation": "trunc_exp",  # can be 'relu' or 'trunc_exp'
    "density_model": "triplane",  # Can be triplane or hexplane
    "density_field_rank": 1,
    "proposal_feature_dim": 10,
    "proposal_decoder_type": "nn",

    # Model settings
    "density_threshold": 1e-4,
    "dmask_update": [-1],  # 1000
    # "upsample_resolution": [3241792, 5832000, 11239424, 16777216],
    # "upsample_steps": [500, 800, 1200, 1500],
    "upsample_F_steps": [],
    "density_multiplier": 1,
    "sh": False,
    "learnedbasis": True,
    "use_F": False,
    # "train_scale_steps": [1000, 2000, 4000],
    "add_rank_steps": [-1],
    "multiscale_res": [1, 2, 4],
    # "feature_len": [64, 64, 64, 64],  # [8, 16, 32, 64] rank 2 is ~2.3 it/s, [4, 8, 16, 32] rank 2 is ~4 it/s, [4, 8, 16, 32] rank 1 is ~7 it/s, [4, 8, 16, 16] rank 1 is ~8 it/s
    # These times might be artificially slower because of sharing a gpu
    "grid_config": """
[
    {
        "input_coordinate_dim": 3,
        "grid_dimensions": 2,
        "output_coordinate_dim": 64,
        "resolution": [64, 64, 64],
        "rank": 1,
    },
    {
        "input_coordinate_dim": 5,
        "resolution": [2, 2, 2, 2, 2],
        "feature_dim": 28,
        "init_std": 0.001
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
