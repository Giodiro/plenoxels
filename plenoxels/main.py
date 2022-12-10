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

from plenoxels.utils import parse_optfloat


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers,
                        force=True)


def load_data(is_video: bool, data_downsample, data_dirs, validate_only: bool, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)

    if is_video:
        from plenoxels.runners import video_trainer
        return video_trainer.load_data(data_downsample, data_dirs, validate_only=validate_only, **kwargs)
    else:
        from plenoxels.runners import multiscene_trainer
        return multiscene_trainer.load_data(data_downsample, data_dirs, validate_only=validate_only, **kwargs)


def init_trainer(is_video: bool, **kwargs):
    if is_video:
        from plenoxels.runners import video_trainer
        return video_trainer.VideoTrainer(**kwargs)
    else:
        from plenoxels.runners import multiscene_trainer
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
    if not validate_only:
        log_dir = os.path.join(config['logdir'], config['expname'])
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
            out.write('config = ' + pprint.pformat(config))

        with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
            for key in config.keys():
                f.write("%s\t%s\n"%(key,config[key]))

    data = load_data(is_video, validate_only=validate_only, **config)
    config.update(data)
    trainer = init_trainer(is_video, **config)
    if args.log_dir is not None:
        checkpoint_path = os.path.join(args.log_dir, "model.pth")
        trainer.load_model(torch.load(checkpoint_path))

    if validate_only:
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
        trainer.validate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
