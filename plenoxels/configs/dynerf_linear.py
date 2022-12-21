# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "salmon_bestlinear",
    "logdir": "./logs/dynerf_linear_refactor1",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 2,
    "data_dirs": ["/home/sfk/data/3DVideo/flame_salmon_1"],
    # Data settings for 360
    "max_train_cameras": None,
    "max_test_cameras": None,
    "max_train_tsteps": None,
    "max_test_tsteps": None,
    # Data settings for LLFF
    "keyframes": False,
    "isg_step": -1,
    "isg": False,
    "ist_step": 80000,
    "contract": False,
    "ndc": True,
    "scene_bbox": [[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]],
    "near_scaling": 0.9,
    "ndc_far": 2.6,

    # Optimization settings
    "num_steps": 120_001,
    "batch_size": 16384,
    "num_batches_per_dset": 1,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 1e-2,

    # Regularization
    "plane_tv_weight": 2e-4,
    "plane_tv_weight_proposal_net": 2e-4,
    "l1_appearance_planes": 1e-4,
    "l1_appearance_planes_proposal_net": 1e-4,
    "time_smoothness_weight": 1e-3,
    "time_smoothness_weight_proposal_net": 1e-5,
    "histogram_loss_weight": 1.0,  # this should be set > 0 when using proposal sampling
    "depth_tv_weight": 0,
    "distortion_loss_weight": 0.001,

    # Training settings
    "train_fp16": True,
    "save_every": 30000,
    "valid_every": 30000,
    "save_outputs": True,

    # Raymarching settings
    "num_samples": 48,
    "single_jitter": False,
    # proposal sampling
    "num_proposal_samples": [256, 96],
    "num_proposal_iterations": 2,
    "use_same_proposal_network": False,
    "proposal_net_args_list": [
        {"resolution": [128, 128, 128, 50], "num_input_coords": 4, "num_output_coords": 8},
        {"resolution": [256, 256, 256, 50], "num_input_coords": 4, "num_output_coords": 8},
    ],

    # Model settings
    "multiscale_res": [1, 2, 4, 8],
    "density_activation": "trunc_exp",
    "linear_decoder": True,
    "grid_config": [{
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 64,
        "grid_dimensions": 2,
        "resolution": [64, 64, 64, 150],
    }],
}
