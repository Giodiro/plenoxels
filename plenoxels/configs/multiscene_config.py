from yacs.config import CfgNode as CN

_C = CN()

_C.seed = 42
_C.expname = "e1"
_C.logdir = "./logs/"
_C.use_csrc = True

_C.sh = CN()
_C.sh.degree = 2

_C.optim = CN()
_C.optim.batch_size = 4000
_C.optim.batches_per_epoch = 500
_C.optim.num_epochs = 10
_C.optim.lr = 1e6
_C.optim.cosine = False
_C.optim.train_fp16 = False

_C.optim.regularization = CN()
_C.optim.regularization.l1_weight = 0.1
_C.optim.regularization.tv_weight = 0.01
_C.optim.regularization.consistency_weight = 0.0

_C.data = CN()
_C.data.datadirs = ["/data/DATASETS/SyntheticNerf/lego", "/data/DATASETS/SyntheticNerf/drums", ]
_C.data.resolution = 256
_C.data.downsample = 1.0
_C.data.max_tr_frames = None
_C.data.max_ts_frames = None

_C.model = CN()
_C.model.num_atoms = [128]
_C.model.coarse_reso = 64
_C.model.fine_reso = [3]
_C.model.noise_std = 0.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
