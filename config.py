from yacs.config import CfgNode as CN

_C = CN()

_C.seed = 42
_C.expname = "hg_exp_5"
_C.logdir = "./logs/"
_C.model_type = "regular_grid"

_C.sh = CN()
_C.sh.degree = 2
_C.sh.sh_encoder = "plenoxels"

_C.optim = CN()
_C.optim.batch_size = 4000
_C.optim.occupancy_penalty = 0.001
_C.optim.profile = False
_C.optim.num_epochs = 10
_C.optim.progress_refresh_rate = 500
_C.optim.eval_refresh_rate = 8000
_C.optim.render_refresh_rate = 20
_C.optim.lr_sigma = None
_C.optim.lr_rgb = None
_C.optim.lr_decay = 10  # 10 * 1000 steps

_C.optim.optimizer = "sgd"

_C.optim.regularization = CN()
_C.optim.regularization.types = ["TV"]
_C.optim.regularization.tv_sh_weight = 0.001
_C.optim.regularization.tv_sigma_weight = 0.00001
_C.optim.regularization.tv_subsample = 100
_C.optim.regularization.sparsity_weight = 0.0001

_C.optim.adam = CN()
_C.optim.adam.lr = 0.5

_C.data = CN()
_C.data.datadir = "/home/giacomo/plenoxels/lego"
_C.data.resolution = 128
_C.data.downsample = 1.0
_C.data.max_tr_frames = 10#None
_C.data.max_ts_frames = 2#None

_C.grid = CN()
_C.grid.ini_rgb = 0.0
_C.grid.ini_sigma = 0.1
_C.grid.update_occ_iters = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
_C.grid.shrink_iters = [2000, 8000]
_C.grid.upsample_iters = [2000, 6000, 36000]
_C.grid.abs_light_thresh = 0.0001
_C.grid.occupancy_thresh = 0.01
_C.grid.reso_multiplier = 1.5

_C.irreg_grid = CN()
_C.irreg_grid.prune_threshold = 0.001
_C.irreg_grid.count_intersections = "tensorrf"
_C.irreg_grid.voxel_mul = 2.0

_C.hash_grid = CN()
_C.hash_grid.log2_hashmap_size = 19


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
