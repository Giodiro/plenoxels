from yacs.config import CfgNode as CN

_C = CN()

_C.seed = 42
_C.expname = "e1"
_C.logdir = "./logs/"

_C.sh = CN()
_C.sh.degree = 0

_C.optim = CN()
_C.optim.batch_size = 4096
_C.optim.lr_decay_iters = -1
_C.optim.lr_init_spatial = 2e-2
_C.optim.lr_init_network = 1e-3
_C.optim.max_steps = 30000
_C.optim.lr_decay_target_ratio = 0.1
_C.optim.upsample_iters = [2000, 3000, 4000, 5500, 7000]

_C.optim.test_every = 1000

_C.optim.regularization = CN()

_C.data = CN()
_C.data.datadir = "/data/DATASETS/SyntheticNerf/lego"
_C.data.resolution = 512
_C.data.downsample = None
_C.data.max_tr_frames = None
_C.data.max_ts_frames = 10

_C.model = CN()
_C.model.reso_init = 128
_C.model.reso_final = 300
_C.model.n_rgb_comp = [48, 48, 48]
_C.model.n_sigma_comp = [16, 16, 16]
_C.model.abs_light_thresh = 1e-4


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
