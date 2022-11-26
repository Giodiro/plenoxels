import argparse
import importlib.util
import logging
import os
import pprint
import sys
from typing import List, Dict, Any

import numpy as np
import wandb
import torch
import torch.utils.data
# import time
# import multiprocessing
from multiprocessing import Process
from plenoxels.runners import video_trainer, multiscene_trainer
from plenoxels.utils import parse_optfloat, parse_optint

# multiprocessing.set_start_method("spawn")

def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)


def load_data(is_video: bool, data_downsample, data_dirs, batch_size, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)
    batch_size = parse_optint(batch_size)

    if is_video:
        return video_trainer.load_data(data_downsample, data_dirs, batch_size=batch_size, **kwargs)
    else:
        return multiscene_trainer.load_data(data_downsample, data_dirs, batch_size=batch_size, **kwargs)


def init_trainer(is_video: bool, **kwargs):
    if is_video:
        return video_trainer.VideoTrainer(**kwargs)
    else:
        return multiscene_trainer.Trainer(**kwargs)
    
    
# def get_freer_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     return np.argmax(memory_available)


def main():
    setup_logging()
    
    #gpu = get_freer_gpu()
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    #print(f'gpu is {gpu}')
    
    # Use the wandb.init() API to generate a background process 
    # to sync and log data as a Weights and Biases run.
    # Optionally provide the name of the project. 
    run = wandb.init()

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)

    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(wandb.config.config_path), wandb.config.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config
    # Process overrides from argparse into config
    # overrides can be passed from the command line as key=value pairs. E.g.
    # python plenoxels/main.py --config-path plenoxels/config/cfg.py max_ts_frames=200
    # note that all values are strings, so code should assume incorrect data-types for anything
    # that's derived from config - and should not a string.
    is_video = "keyframes" in config

    pprint.pprint(config)
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
        for key in config.keys():
            f.write("%s\t%s\n"%(key,config[key]))

    if is_video:
        state = None
        trainer, config = video_trainer.load_video_model(config, state, validate_only=False)
    else:
        data = load_data(is_video, **config)
        config.update(data)
        trainer: multiscene_trainer.Trainer = init_trainer(is_video, **config)

    trainer.train()


    
    # # Import config
    # spec = importlib.util.spec_from_file_location(os.path.basename(wandb.config.config_path), wandb.config.config_path)
    # #spec = importlib.util.spec_from_file_location("plenoxels/config/giac_learnablehash.py")
    # cfg = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(cfg)
    # config: Dict[str, Any] = cfg.config
    # print("config", config)
    # print("wandb.config", wandb.config)
    # # note that we define values from `wandb.config` instead of 
    # # defining hard values
    # for k, v in wandb.config.items():
    #     if k != 'config_path':
    #         # ensure that all values we tune are in the config
    #         assert k in config
            
    #         config[k] = v
    
    # pprint.pprint(config)
    
    # log_dir = os.path.join(config['logdir'], config['expname'])
    # os.makedirs(log_dir, exist_ok=True)
    # with open(os.path.join(log_dir, 'config.py'), 'wt') as out: 
    #     out.write('config = ' + pprint.pformat(config))

    # with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
    #     for key in config.keys():
    #         f.write("%s\t%s\n"%(key,config[key]))
    
    # data = load_data(False, **config)
    # config.update(data)
    # trainer: multiscene_trainer.Trainer = init_trainer(False, **config)
    # trainer.train()


if __name__ == "__main__":
    main()
    # sweep_configuration = {
    #     'method': 'bayes',
    #     'name': 'ficus_sweep',
    #     'metric': {
    #         'goal': 'maximize', 
    #         'name': 'test_psnr'
    #         },
    #     'parameters': {
    #         'lr': {'max': 0.1, 'min': 0.0001},
    #         'plane_tv_weight_sigma': {'max': 0.1, 'min': 0.0},
    #         'plane_tv_weight_sh': {'max': 0.1, 'min': 0.0},
    #         "scheduler_type": {'values': [None, "warmup_log_linear", "warmup_cosine", "warmup_step_many"]},
    #         'config_path': {'values': ['plenoxels/configs/ficus_config_sara.py']},
    #     }
    # }
        
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="hexplane")
    # print(sweep_id)
    # wandb.agent(sweep_id, function=main, count=4)
    
    # run the sweep
    # agents = 4

    # procs = []
    # for process in range(agents):
        
    #     proc = Process(target=wandb.agent, args=(sweep_id, main, ))
    #     proc.start()
    #     procs.append(proc)
        
    #     #time.sleep(30)

    # # # complete the processes
    # for proc in procs:
    #     proc.join()
    
