# configuration file to be used with `main.py` for normal (or multiscene) training
# the configuration must be specified in a dictionary called `config`.
config = {
    "expname": "contract_sf2_lr1e-2_applr5e-2_ptv1e-4_ptvpn1e-4_dl0",
    "logdir": "./logs/trevi",
    "device": "cuda:0",

    # Data settings
    "data_downsample": 1,
    "data_dirs": ["/data/DATASETS/phototourism/trevi-fountain"],
    "contract": False,
    "ndc": False,
    "scene_bbox": [[-2.0, -1.0, -2.0], [2.0, 1.0, 2.0]],
    "scale_factor": 6.,
    "orientation_method": "none",
    "center_poses": True,
    "auto_scale_poses": True,
    "near_scaling": 0.9,  # unused
    "ndc_far": 2.6,       # unused

    # Optimization settings
    "num_steps": 60_001,
    "batch_size": 4096,
    "scheduler_type": "warmup_cosine",
    "optim_type": "adam",
    "lr": 1e-2,
    # test latent code optimization
    "app_optim_n_epochs": 5,
    "app_optim_lr": 1e-1,

    # Regularization
    "plane_tv_weight": 1e-4,
    "plane_tv_weight_proposal_net": 1e-4,
    # "l1_appearance_planes": 1e-4,
    # "l1_appearance_planes_proposal_net": 1e-4,
    "histogram_loss_weight": 1.0,  # this should be set > 0 when using proposal sampling
    "depth_tv_weight": 0,
    "distortion_loss_weight": 1e-4,

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
        {"resolution": [64, 64, 64], "num_input_coords": 3, "num_output_coords": 8},
        {"resolution": [128, 128, 128], "num_input_coords": 3, "num_output_coords": 8},
    ],

    # Model settings
    "multiscale_res": [2, 4, 8],
    "density_activation": "trunc_exp",
    "appearance_embedding_dim": 24,
    "concat_features_across_scales": True,
    "linear_decoder": False,
    "grid_config": [{
        "input_coordinate_dim": 3,
        "output_coordinate_dim": 16,
        "grid_dimensions": 2,
        "resolution": [86, 56, 52],
    }],
}
