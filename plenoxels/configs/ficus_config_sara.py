config = {'add_rank_steps': [-1],
 'batch_size': 4096,
 'data_dirs': ["/home/sfk/data/nerf_synthetic/ficus"],
 'data_downsample': 1,
 'data_resolution': None,
 'density_activation': 'trunc_exp',
 'density_field_rank': 10,
 'density_field_resolution': [128, 256],
 'density_model': 'triplane',
 'density_multiplier': 1,
 'density_threshold': 0.0001,
 'dmask_update': [100000],
 'expname': 'ficus_noF_nn',
 'floater_loss': 0,
 'grid_config': '\n'
                '[\n'
                '    {\n'
                '        "input_coordinate_dim": 3,\n'
                '        "output_coordinate_dim": 28,\n'
                '        "grid_dimensions": 2,\n'
                '        "resolution": [64, 64, 64],\n'
                '        "rank": 2,\n'
                '    },\n'
                '    {\n'
                '        "input_coordinate_dim": 5,\n'
                '        "resolution": [2, 2, 2, 2, 2],\n'
                '        "feature_dim": 28,\n'
                '        "init_std": 0.001\n'
                '    }\n'
                ']\n',
 'histogram_loss_weight': 1,
 'hold_every': 8,
 'l1density_weight': 0,
 'logdir': './logs/sh/ficus',
 'lr': 0.01,
 'max_tr_frames': None,
 'max_ts_frames': 10,
 'multiscale_res': [1, 2, 4, 8],
 'n_intersections': 48,
 'num_batches_per_dset': 1,
 'num_proposal_samples': [256, 96],
 'num_steps': 30_001,
 'optim_type': 'adam',
 'plane_tv_weight_sigma': 0.0,
 'plane_tv_weight_sh': 0.0,
 'raymarch_type': 'fixed',
 'regnerf_weight_end': 0.0,
 'regnerf_weight_max_step': 512,
 'regnerf_weight_start': 0,
 'save_every': 30_000,
 'save_outputs': True,
 'scheduler_type': None,
 'sh': False,
 'single_jitter': False,
 'spacing_fn': 'linear',
 'train_fp16': True,
 'transfer_learning': False,
 'upsample_F_steps': [],
 'use_F': False,
 'use_trainable_rank': False,
 'valid_every': 30_000, 
 'volume_tv_npts': 1024,
 'volume_tv_weight': 0.0}