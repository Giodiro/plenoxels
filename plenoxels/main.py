import argparse
import importlib.util
import logging
import os
import pprint
import sys
from typing import List, Dict, Any

import numpy as np


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')

import torch
import torch.utils.data

from plenoxels.runners import video_trainer, multiscene_trainer
from plenoxels.runners.utils import get_freer_gpu
from plenoxels.utils import parse_optfloat, parse_optint


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


def main():
    setup_logging()

    p = argparse.ArgumentParser(description="")

    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('override', nargs=argparse.REMAINDER)

    args = p.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config
    # Process overrides from argparse into config
    # overrides can be passed from the command line as key=value pairs. E.g.
    # python plenoxels/main.py --config-path plenoxels/config/cfg.py max_ts_frames=200
    # note that all values are strings, so code should assume incorrect data-types for anything
    # that's derived from config - and should not a string.
    overrides: List[str] = args.override
    overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}
    config.update(overrides_dict)
    is_video = "keyframes" in config
    validate_only = args.validate_only

    pprint.pprint(config)
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.path.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.txt'), 'wt') as out:
        pprint.pprint(config, stream=out)

    if is_video:
        state = None
        if args.log_dir is not None:
            checkpoint_path = os.path.join(args.log_dir, "model.pth")
            state = torch.load(checkpoint_path)
        trainer, config = video_trainer.load_video_model(config, state, validate_only)
    else:
        data = load_data(is_video, **config)
        config.update(data)
        trainer: multiscene_trainer.Trainer = init_trainer(is_video, **config)
        if trainer.transfer_learning:
            # We have reloaded the model learned from args.log_dir
            assert args.log_dir is not None and os.path.isdir(args.log_dir)
        if args.log_dir is not None:
            checkpoint_path = os.path.join(args.log_dir, "model.pth")
            trainer.load_model(torch.load(checkpoint_path))

    if args.validate_only:
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
        trainer.validate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
