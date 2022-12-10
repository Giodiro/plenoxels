config = {
    'expname': 'jumpingjacks_linear_d64noconcat_occ128_flatlr10k',
    'logdir': './logs/dnerf',
    'device': 'cuda:0',
    'wandb': False,

    # Data settings
    'data_dirs': ['/home/sfk/data/dnerf/data/jumpingjacks'],
    'data_downsample': 1.0,
    'max_test_cameras': None,
    'max_test_tsteps': None,
    'max_train_cameras': None,
    'max_train_tsteps': None,
    # Sampling weights
    'isg': False,
    'isg_step': -1,
    'ist_step': -1,
    'keyframes': False,

    # Grid config
    'grid_config': [
        {
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 64,
            "grid_dimensions": 2,
            "resolution": [64, 64, 64, 100],
        }
    ],

    # Optimization settings
    'num_steps': 10001,
    'optim_type': 'adam',
    'scheduler_type': 'None',
    'lr': 0.005,

    # Training settings
    'train_fp16': True,
    'save_every': 10000,
    'valid_every': 10000,
    'save_outputs': True,

    # Regularization
    'l1_appearance_planes_reg': 0.001,
    'l1_plane_color_reg': 0,
    'l1_plane_density_reg': 0,
    'l1density_weight': 0,
    'time_smoothness_weight': 0.01,
    'plane_tv_weight': 0.01,

    # Raymarching
    'cone_angle': 0.0,
    'n_samples': 1024,
    'sample_batch_size': 1 << 18,
    'early_stop_eps': 1e-4,
    'alpha_threshold': 1e-3,
    'density_threshold': 1e-2,

    # Model
    'sh': False,
    'learnedbasis': True,
    'use_F': False,
    'concat_features': False,
    'train_every_scale': False,
    'density_activation': 'trunc_exp',
    'multiscale_res': [1, 2, 3, 4],
    'occupancy_grid_resolution': [128, 128, 128],
}
