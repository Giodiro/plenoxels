# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "orchids_ndc_far2.5_ptv2e-4_propnetptv2e-4_cosine_lr2e-2_nearscale0.9_dl0.01",
    "logdir": "./logs",
    "device": "cuda:0",

    # Data settings
    "data_resolution": None,
    "data_downsample": 4,
    #"data_dirs": ["/data/DATASETS/SyntheticNerf/ficus"],
    "data_dirs": ["/data/DATASETS/LLFF/orchids"],
    # Data settings for 360
    "max_tr_frames": 100,
    "max_ts_frames": 50,
    # Data settings for LLFF
    "hold_every": 8,
    "contract": False,
    "ndc": True,

    # Optimization settings
    "num_steps": 30_001,
    "batch_size": 4096,
    "eval_batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 2e-2,

    # Regularization
    "plane_tv_weight": 2e-4,
    "plane_tv_weight_proposal_net": 2e-4,
    "l1_proposal_net_weight": 0,
    "histogram_loss_weight": 1.0,  # this should be set > 0 when using proposal sampling
    "depth_tv_weight": 0,
    "distortion_loss_weight": 0.01,

    # Training settings
    "train_fp16": True,
    "save_every": 10000,
    "valid_every": 10000,
    "save_outputs": True,

    # Raymarching settings
    "num_samples": 48,
    "single_jitter": False,
    # proposal sampling
    "num_proposal_samples": [256, 128],
    "num_proposal_iterations": 2,
    "use_same_proposal_network": False,
    "proposal_net_args_list": [
        {"resolution": [128, 128, 128], "num_input_coords": 3, "num_output_coords": 8},
        {"resolution": [256, 256, 256], "num_input_coords": 3, "num_output_coords": 8},
    ],

    # Model settings
    "multiscale_res": [1, 2, 4, 8],
    "density_activation": "trunc_exp",
    "grid_config": [{
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 16,
        "grid_dimensions": 2,
        "resolution": [64, 64, 64],
    }],
}
