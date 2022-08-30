import logging
import sys

import torch

from .runners.multiview_runner import MultiviewTrainer

args = {
    "trainer_type": "MultiviewTrainer",
    "exp_name": "test",
    "log_level": 20,
    "grid_type": "LearnableHashGrid",
    "grid_config": """
[
    [
        {
            "input_coordinate_dim": 3,
            "output_coordinate_dim": 3,
            "grid_dimensions": 3,
            "resolution": 128,
            "rank": 1,
            "init_std": 0.01,
        },
        {
            "input_coordinate_dim": 3,
            "resolution": 8,
            "feature_dim": 32,
            "init_std": 0.05
        }
    ]
]
""",
    "multiscale_type": "sum",
    "num_lods": 1,

    "decoder_type": "ingp",
    "decoder_lod_idx": 0,
    "sigma_net_width": 64,
    "sigma_net_layers": 1,
    "sh_degree": 2,

    "dataset_type": "multiview",
    #"dataset_paths": ["/data/DATASETS/SyntheticNerf/lego"],
    "dataset_paths": ["/data/DATASETS/Nerf/fox"],
    "dataset_num_workers": 4,
    "multiview_dataset_format": "standard",
    # "num_rays_sampled_per_img": 4096,
    "bg_color": "white",
    "data_resize_shape": (400, 400),

    "optimizer_type": "adam",
    "lr": 2e-3,
    "lr_scheduler_type": "no-schedule",
    "weight_decay": 0,
    "grid_lr_weight": 1.0,

    "num_epochs": 50,
    "batch_size": 4096,
    "save_every": 10,
    "log_dir": "./logs",
    "random_lod": False,
    "valid_every": 10,
    "pretrained": False,
    "valid_only": False,

    "render_batch": 4000,
    "num_steps": 256,
}


def load_modules(cfg):
    import torch
    from .models.nerf import NeuralRadianceField
    from .tracers import PackedRFTracer
    from .datasets import MultiviewDataset, SampleRays
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nef_cls = NeuralRadianceField
    tracer_cls = PackedRFTracer

    nef = nef_cls(**cfg)
    tracer = tracer_cls(nef=nef, **cfg)
    tracer.to(device)

    if cfg["dataset_type"] == "multiview":
        transform = None#SampleRays(cfg["num_rays_sampled_per_img"])
        train_datasets = []
        for dset in cfg["dataset_paths"]:
            train_datasets.append(MultiviewDataset(dataset_path=dset, transform=transform, **cfg))
            train_datasets[-1].init()
    else:
        raise ValueError(f"Dataset type {cfg['dataset_type']} invalid.")

    return tracer, train_datasets, device


def load_optimizer(cfg):
    """Utility function to get the optimizer from the parsed config.
    """
    str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}
    optim_cls = str2optim[cfg["optimizer_type"]]
    if cfg["optimizer_type"] == 'adam':
        optim_params = {'eps': 1e-15}
    elif cfg["optimizer_type"] == 'sgd':
        optim_params = {'momentum': 0.8}
    else:
        optim_params = {}
    return optim_cls, optim_params


def setup_logging(cfg):
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=cfg["log_level"],
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)


def start(cfg):
    cfg["num_scenes"] = len(cfg["dataset_paths"])
    setup_logging(cfg)
    tracer, train_datasets, device = load_modules(cfg)
    optim_cls, optim_params = load_optimizer(cfg)
    trainer_cls = globals()[cfg["trainer_type"]]
    trainer = trainer_cls(
        tracer=tracer, datasets=train_datasets, optim_cls=optim_cls, optim_params=optim_params,
        device=device, **cfg)
    if cfg["pretrained"]:
        trainer.load_model(torch.load(cfg["pretrained"]))
    if cfg["valid_only"]:
        trainer.validate()
    else:
        trainer.train()


if __name__ == "__main__":
    start(args)
