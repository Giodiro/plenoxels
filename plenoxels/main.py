import argparse
import importlib.util
import logging
import os
import pprint
import sys

import numpy as np

from plenoxels.runners.multiscene_trainer import Trainer
from plenoxels.runners.video_trainer import VideoTrainer

np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.utils.data

from plenoxels.runners.utils import *
from plenoxels.datasets.synthetic_nerf_dataset import SyntheticNerfDataset
from plenoxels.datasets.llff_dataset import LLFFDataset
from plenoxels.datasets.llff_video_dataset import VideoDataset


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
    if data_downsample is None:
        data_downsample = 1.0
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
            assert len(data_dirs) == 1, "Video-datasets don't support multiple training-scenes"
            logging.info(f"About to load data with downsample={data_downsample} and using "
                         f"{kwargs.get('subsample_time_train') * 100}% of the video frames")
            tr_dsets.append(VideoDataset(
                data_dir, split='train', downsample=data_downsample,
                subsample_time=kwargs.get('subsample_time_train')))
            ts_dsets.append(VideoDataset(
                data_dir, split='test', downsample=data_downsample,
                subsample_time=kwargs.get('subsample_time_train')))
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

    args = p.parse_args()
    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    pprint.pprint(cfg.config)
    dset_type, tr_loaders, ts_dsets = load_data(**cfg.config)
    trainer: Trainer = init_trainer(dset_type, tr_loaders, ts_dsets, **cfg.config)
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
