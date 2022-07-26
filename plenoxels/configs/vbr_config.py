from yacs.config import CfgNode as CN

_C = CN()

_C.seed = 42
_C.expname = "vbr_e1"
_C.logdir = "./logs/"

_C.data = CN()
_C.data.datadir = "/data/DATASETS/SyntheticNerf/lego"
_C.data.resolution = None
_C.data.downsample = None
_C.data.max_tr_frames = None
_C.data.max_ts_frames = None


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
