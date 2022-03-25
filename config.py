from yacs.config import CfgNode as CN

_C = CN()

_C.seed = 42
_C.expname = "hg_exp_1"
_C.logdir = "./logs/"
_C.model_type = "hash_grid"

_C.sh = CN()
_C.sh.degree = 2
_C.sh.sh_encoder = "plenoxels"

_C.optim = CN()
_C.optim.batch_size = 4000
_C.optim.occupancy_penalty = 0.0
_C.optim.lr_sigma = None
_C.optim.lr_rgb = None
_C.optim.profile = False
_C.optim.num_epochs = 10
_C.optim.progress_refresh_rate = 50
_C.optim.eval_refresh_rate = 1000
_C.optim.render_refresh_rate = 5

_C.data = CN()
_C.data.datadir = "/home/giacomo/plenoxels/lego"
_C.data.resolution = 256
_C.data.downsample = 1.0
_C.data.max_tr_frames = None
_C.data.max_ts_frames = 10

_C.grid = CN()
_C.grid.ini_rgb = 0.0
_C.grid.ini_sigma = 0.1

_C.irreg_grid = CN()
_C.irreg_grid.prune_threshold = 0.001
_C.irreg_grid.count_intersections = "plenoxels"

_C.hash_grid = CN()
_C.hash_grid.log2_hashmap_size = 19


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
