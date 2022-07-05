from yacs.config import CfgNode as CN

_C = CN()

_C.seed = 42
_C.expname = "dict256_4.128"
_C.reload = "plenoxel_lego_256_v2"
_C.logdir = "./logs/"

_C.optim = CN()
_C.optim.batch_size = 400
_C.optim.num_batches = 100000
_C.optim.lr = 1e-4

_C.model = CN()
_C.model.num_atoms = 128
_C.model.patch_reso = 8


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
