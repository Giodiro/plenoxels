config = {'batch_size': 4096,                                                                                                                                                                                                                          
 'data_dirs': ["/home/sfk/data/nerf_synthetic/ficus"],                                                                                                                                                                                         
 'data_downsample': 1,                                                                                                                                                                                                                        
 'data_resolution': None,                                                                                                                                                                                                                     
 'density_activation': 'trunc_exp',                                                                                                                                                                                                           
 'density_field_rank': 10,                                                                                                                                                                                                                    
 'density_field_resolution': [128, 256],                       
 'density_multiplier': 1,                                                                                                                                                                                                                     
 'density_threshold': 0.0004,    
 'dmask_update': [],                                                                                       
 'expname': 'ficus_test_propsampling_wF',                                                                                                                                                                                                     
 'floater_loss': 0,                                                                                                                                                                                                                           
 'grid_config': '\n'                                           
                '[\n'                                                                                                                                                                                                                         
                '    {\n'                                                                                                                                                                                                                     
                '        "input_coordinate_dim": 3,\n'                                                                                                                                                                                        
                '        "output_coordinate_dim": 5,\n'                                                                                                                                                                                       
                '        "grid_dimensions": 2,\n'
                '        "resolution": [64, 64, 64],#[256, 192, 168],\n'
                '        "rank": 4,\n'
                '    },\n'
                '    {\n'
                '        "input_coordinate_dim": 5,\n'
                '        "resolution": [6, 6, 6, 6, 6],\n'
                '        "feature_dim": 28,\n'
                '        "init_std": 0.001,\n'
                '    }\n'
                ']\n',
 'histogram_loss_weight': 1.0,
 'hold_every': 8,
 'l1_plane_color_weight': 0.0,
 'l1_plane_density_weight': 0.0,
 'l1density_weight': 0.0,
 'logdir': './logs',
 'lr': 0.02,
 'max_tr_frames': 100,
 'max_ts_frames': 10,
 'multiscale_res': [1, 2, 4, 8],
 'n_intersections': 48,
 'num_batches_per_dset': 1,
 'num_proposal_samples': [256, 96],
 'num_steps': 30001,
 'optim_type': 'adam',
 'plane_tv_weight': 0.0001,
#  'plane_tv_weight_sigma': 0.0,
#  'plane_tv_weight_sh': 0.01,
 'plane_tv_what': 'Gcoords',
 'raymarch_type': 'fixed',
 'save_every': 30000,
 'save_outputs': True,
 'scheduler_type': 'warmup_step_many',
 'sh': True,
 'sh_decoder_type': 'manual',
 'single_jitter': False,
 'train_fp16': True,
 'transfer_learning': False,                                                                                                                                                                                                                  
 'upsample_resolution': [],                                                                                                                                                                                                                   
 'upsample_steps': [],                                                                                                                                                                                                                        
 'use_F': True,                                                                                                                                                                                                                               
 'valid_every': 30000,                                                                                                                                                                                                                         
 'volume_tv_npts': 100,                                                                                                                                                                                                                       
 'volume_tv_patch_size': 8,                                    
 'volume_tv_weight': 0.0,                                                                                                                                                                                                                     
 'volume_tv_what': 'Gcoords'}