from yacs.config import CfgNode as CN

_C = CN()

_C.seed = 42
_C.expname = "e1"
_C.logdir = "./logs/"

_C.sh = CN()
_C.sh.degree = 2

_C.optim = CN()
_C.optim.batch_size = 4000
_C.optim.num_epochs = 10
_C.optim.lr = 1e-2
_C.optim.cosine = True

_C.optim.regularization = CN()
_C.optim.regularization.commitment_cost = 0.1

_C.data = CN()
_C.data.datadir = "/data/DATASETS/SyntheticNerf/lego"
_C.data.resolution = None
_C.data.downsample = None
_C.data.max_tr_frames = None
_C.data.max_ts_frames = None

_C.model = CN()
_C.model.resolution = 32
_C.model.coarse_dim = 64
_C.model.num_embeddings = 1024
_C.model.num_freqs_pt = 6
_C.model.num_freqs_dir = 4


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
