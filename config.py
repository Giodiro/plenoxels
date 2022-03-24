from yacs.config import CfgNode as CN

_C = CN()

_C.expname = "an_exp"

_C.sh = CN()
_C.sh.degree = 2
_C.sh.sh_encoder = "plenoxel"

_C.optim = CN()
_C.optim.batch_size = 4000
_C.optim.occupancy_penalty = 0.0
_C.optim.lr_sigma = None
_C.optim.lr_rgb = None
_C.optim.profile = False
_C.optim.num_epochs = 10
_C.optim.progress_refresh_rate = 5

_C.data = CN()
_C.data.datadir = "/home/giacomo/plenoxels/lego"
_C.data.resolution = 256
_C.data.downsample = 1.0

_C.grid = CN()
_C.grid.ini_rgb = 0.0
_C.grid.ini_sigma = 0.1

_C.irreg_grid = CN()
_C.irreg_grid.prune_threshold = 0.001
_C.irreg_grid.count_intersections = "plenoxel"

_C.random_seed = 42

_C.policy = CN()
_C.policy.name = "deterministic_v1"
_C.policy.num_evals = 1

_C.dataset = CN()
_C.dataset.name = "cifar10"
_C.dataset.subsample_train = 0.1
_C.dataset.subsample_test = 0.1

_C.kernel = CN()
_C.kernel.tr_kernel_path = ""
_C.kernel.ts_kernel_path = ""
_C.kernel.sigma = 3.0            # This is ignored is a kernel_path is specified
_C.kernel.dtype = "float32"

_C.solve = CN()
# A list of regularization strengths to try out
_C.solve.lambdas = [1e-8]
_C.solve.algo = "dual"
_C.solve.dtype = "float64"
_C.solve.output_path = ""

_C.nystrom = CN()
_C.nystrom.num_centers = 5000
_C.nystrom.regularizer = 1e-7
_C.nystrom.maxiter = 10
_C.nystrom.dtype = "float32"
_C.nystrom.sigma = [3.0, 30.0, 300.0]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
