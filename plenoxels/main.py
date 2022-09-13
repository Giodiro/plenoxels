import argparse
import importlib.util
import logging
import os
import pprint
import sys
from typing import List, Dict, Any, Optional

import numpy as np

from plenoxels.runners.multiscene_trainer import Trainer
from plenoxels.runners.video_trainer import VideoTrainer

import torch
import torch.utils.data

from plenoxels.runners.utils import *
from plenoxels.datasets.synthetic_nerf_dataset import SyntheticNerfDataset
from plenoxels.datasets.llff_dataset import LLFFDataset
from plenoxels.datasets.llff_video_dataset import VideoDataset


def parse_optfloat(val, default_val=None) -> Optional[float]:
    if val == "None" or val is None:
        return default_val
    return float(val)


def parse_optint(val, default_val=None) -> Optional[int]:
    if val == "None" or val is None:
        return default_val
    return int(val)


def setup_logging(log_level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)


def decide_dset_type(data_dir) -> str:
    if ("chair" in data_dir or "drums" in data_dir or "ficus" in data_dir or "hotdog" in data_dir
            or "lego" in data_dir or "materials" in data_dir or "mic" in data_dir
            or "ship" in data_dir):
        return "synthetic"
    elif ("fern" in data_dir or "flower" in data_dir or "fortress" in data_dir
          or "horns" in data_dir or "leaves" in data_dir or "orchids" in data_dir
          or "room" in data_dir or "trex" in data_dir):
        return "llff"
    elif ("coffee_martini" in data_dir or "cut_roasted_beef" in data_dir):
        return "video"
    else:
        raise RuntimeError(f"data_dir {data_dir} not recognized as LLFF, Synthetic or Video dataset.")


def load_data(data_resolution, data_downsample, data_dirs, max_tr_frames, max_ts_frames, hold_every, batch_size, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)
    data_resolution = parse_optint(data_resolution)
    max_tr_frames = parse_optint(max_tr_frames)
    max_ts_frames = parse_optint(max_ts_frames)
    hold_every = parse_optint(hold_every)
    batch_size = parse_optint(batch_size)
    # Training datasets are lists of lists, where each inner list is different resolutions for the same scene
    # Test datasets are a single list over the different scenes, all at full resolution
    tr_dsets, tr_loaders, ts_dsets = [], [], []
    for data_dir in data_dirs:
        dset_type = decide_dset_type(data_dir)
        # TODO: multiple different dataset types are currently not supported well.
        if dset_type == "synthetic":
            logging.info(f"About to load data at reso={data_resolution}, downsample={data_downsample}")
            tr_dsets.append(SyntheticNerfDataset(
                data_dir, split='train', downsample=data_downsample, resolution=data_resolution,
                max_frames=max_tr_frames))
            ts_dsets.append(SyntheticNerfDataset(
                data_dir, split='test', downsample=1, resolution=800, max_frames=max_ts_frames))
        elif dset_type == "llff":
            logging.info(f"About to load data at reso={data_resolution}, downsample={data_downsample}")
            tr_dsets.append(LLFFDataset(
                data_dir, split='train', downsample=data_downsample, resolution=data_resolution,
                hold_every=hold_every))
            ts_dsets.append(LLFFDataset(
                data_dir, split='test', downsample=1, resolution=None, hold_every=hold_every))
        elif dset_type == "video":
            subsample_time_train = float(kwargs.get('subsample_time_train'))
            assert len(data_dirs) == 1, "Video-datasets don't support multiple training-scenes"
            logging.info(f"About to load data with downsample={data_downsample} and using "
                         f"{subsample_time_train * 100}% of the video frames")
            tr_dsets.append(VideoDataset(
                data_dir, split='train', downsample=data_downsample,
                subsample_time=subsample_time_train))
            ts_dsets.append(VideoDataset(
                data_dir, split='test', downsample=data_downsample,
                subsample_time=subsample_time_train))
        else:
            raise ValueError(dset_type)
        tr_loaders.append(torch.utils.data.DataLoader(
            tr_dsets[-1], batch_size=batch_size, shuffle=True, num_workers=3, prefetch_factor=4,
            pin_memory=True))

    return dset_type, tr_loaders, ts_dsets


def init_trainer(dset_type, tr_loaders, ts_dsets, **kwargs):
    if dset_type == "video":
        return VideoTrainer(tr_loaders=tr_loaders, ts_dsets=ts_dsets, **kwargs)
    else:
        return Trainer(tr_loaders=tr_loaders, ts_dsets=ts_dsets, **kwargs)


def main():
    setup_logging()
    gpu = get_freer_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logging.info(f"Selected GPU {gpu}")

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

    pprint.pprint(config)
    dset_type, tr_loaders, ts_dsets = load_data(**config)
    trainer: Trainer = init_trainer(dset_type, tr_loaders, ts_dsets, **config)
    if trainer.transfer_learning:
        # We have reloaded the model learned from args.log_dir
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
    if args.log_dir is not None:
        trainer.log_dir = args.log_dir
        checkpoint_path = os.path.join(trainer.log_dir, "model.pth")
        trainer.load_model(torch.load(checkpoint_path))

    if args.validate_only:
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
        trainer.validate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
