# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "flame_salmon_ndc",
    "logdir": "./logs/flame_salmon",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 2,
    "data_dirs": ["/data/DATASETS/VidNerf/flame_salmon"],
    # Data settings for 360
    "max_train_cameras": 25,
    "max_test_cameras": 1,
    "max_train_tsteps": None,
    "max_test_tsteps": None,
    "max_tr_frames": 100,
    "max_ts_frames": 50,
    # Data settings for LLFF
    "keyframes": False,
    "isg_step": -1,
    "isg": False,
    "ist_step": 80000,
    "contract": False,
    "ndc": True,

    # Optimization settings
    "num_steps": 30_001,
    "batch_size": 4096,
    "num_batches_per_dset": 1,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 2e-2,

    # Regularization
    "plane_tv_weight": 2e-4,
    "plane_tv_weight_proposal_net": 2e-4,
    "l1_appearance_planes": 1e-2,
    "l1_appearance_planes_proposal_net": 1e-2,
    "time_smoothness_weight": 0.1,
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
        {"resolution": [128, 128, 128, 50], "num_input_coords": 4, "num_output_coords": 8},
        {"resolution": [256, 256, 256, 50], "num_input_coords": 4, "num_output_coords": 8},
    ],

    # Model settings
    "multiscale_res": [1, 2, 4, 8],
    "density_activation": "trunc_exp",
    "grid_config": [{
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 16,
        "grid_dimensions": 2,
        "resolution": [64, 64, 64, 150],
    }],
}
