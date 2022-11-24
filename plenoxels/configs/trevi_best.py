config = {'add_rank_steps': [],
 'batch_size': 4096,
 'data_dirs': ['/home/warburg/data/phototourism/trevi'],
 'data_downsample': 1.0,
 'density_activation': 'trunc_exp',
 'density_field_rank': 1,
 'density_field_resolution': [128, 256],
 'density_model': 'triplane',
 'expname': 'trevi_cvpr',
 'floater_loss': 0.0,
 'grid_config': '\n'
                '[\n'
                '    {\n'
                '        "input_coordinate_dim": 3,\n'
                '        "output_coordinate_dim": 32,\n'
                '        "grid_dimensions": 2,\n'
                '        "resolution": [80, 40, 20], \n'
                '        "rank": 2,\n'
                '        "time_reso": 1708,\n'  # 1708 for trevi, 1200 for sacre, 773 for brandenburg
                '    },\n'
                '    {\n'
                '        "input_coordinate_dim": 2,\n'
                '        "resolution": [6, 6, 6, 6, 6],\n'
                '        "feature_dim": 32,\n'
                '        "init_std": 0.001\n'
                '    }\n'
                ']\n',
 'histogram_loss_weight': 1,
 'isg': False,
 'isg_step': -1,
 'ist_step': -1,
 'keyframes': False,
 'l1_appearance_planes_reg': 0.0,
 'l1density_weight': 0,
 'logdir': './logs/phototourism',
 'lr': 0.01,
 'multiscale_res': [1, 2, 4, 8],
 'n_intersections': 64,
 'num_proposal_samples': [256, 96],
 'num_sample_multiplier': 2,
 'num_steps': 60001,
 'optim_type': 'adam',
 'plane_tv_weight': 0,
 'proposal_decoder_type': 'nn',
 'proposal_feature_dim': 10,
 'raymarch_type': 'fixed',
 'regnerf_weight_end': 0.0,
 'regnerf_weight_max_step': 512,
 'regnerf_weight_start': 0,
 'save_every': 60000,
 'save_outputs': True,
 'save_video': True,
 'scheduler_type': 'warmup_cosine',
 'sh': False,
 'single_jitter': False,
 'spacing_fn': 'linear',
 'train_fp16': True,
 'upsample_time_resolution': [],
 'upsample_time_steps': [],
 'use_F': False,
 'valid_every': 60000,
 'volume_tv_npts': 1024,
 'volume_tv_weight': 0.0,
 'volume_tv_what': 'Gcoords'}