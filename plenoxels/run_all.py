from plenoxels.main import load_data, init_trainer, save_config, setup_logging

from typing import List, Dict, Any
import os
import pprint
from copy import copy

import numpy as np

import torch
import torch.utils.data


LLFF_DSETS = ["fern", "orchids", "trex", "room", "leaves", "fortress", "flower", "horns"]
LLFF_DATADIR = "/data/DATASETS/LLFF"
SYNTH360_DSETS = ["chair", "ficus", "drums", "hotdog", "lego", "materials", "mic", "ship"]
SYNTH360_DATADIR = "/data/DATASETS/SyntheticNerf"
DYNERF_DSETS = ["coffee_martini", "flame_steak", "sear_steak", "cook_spinach", "cut_roasted_beef", "flame_salmon"]
DYNERF_DATADIR = "/data/DATASETS/VidNerf"


def main():
    setup_logging()
    seed = 8
    np.random.seed(seed)
    torch.manual_seed(seed)

    exp_type = "dynerf"
    base_config: Dict[str, Any]
    datasets: List[str]
    datadir: str
    if exp_type == "dynerf":
        #import plenoxels.configs.test_flamesalmon as dynerf_config
        import plenoxels.configs.dynerf_linear as dynerf_config
        base_config = dynerf_config.config
        datasets = DYNERF_DSETS
        datadir = DYNERF_DATADIR
    elif exp_type == "synthetic360":
        import plenoxels.configs.giac_learnablehash as static_config
        base_config = static_config.config
        datasets = SYNTH360_DSETS
        datadir = SYNTH360_DATADIR
    elif exp_type == "llff":
        import plenoxels.configs.giac_learnablehash as llff_config
        base_config = llff_config.config
        datasets = LLFF_DSETS
        datadir = LLFF_DATADIR
    else:
        raise ValueError()

    base_expname = base_config['expname']

    for dataset in datasets:
        config = copy(base_config)
        config['data_dirs'][0] = os.path.join(datadir, dataset)
        config['expname'] = f'{dataset}_{base_expname}'

        pprint.pprint(config)
        save_config(config)
        data = load_data(model_type="video", validate_only=False, render_only=False, **config)
        config.update(data)
        trainer = init_trainer(model_type="video", **config)
        trainer.train()


if __name__ == "__main__":
    main()
