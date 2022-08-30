from importlib.machinery import PathFinder
from pathlib import Path
from typing import Optional

import torch

spec = PathFinder().find_spec("c_ext", [str(Path(__file__).resolve().parents[1])])
torch.ops.load_library(spec.origin)


def grid_sample_4d(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if mode == "bilinear":
        interpolation_mode = 0
    elif mode == "nearest":
        interpolation_mode = 1
    elif mode == "bicubic":
        interpolation_mode = 2
    else:
        raise ValueError(mode)
    if padding_mode == "zeros":
        padding_mode_ = 0
    elif padding_mode == "border":
        padding_mode_ = 1
    elif padding_mode == "reflection":
        padding_mode_ = 2
    else:
        raise ValueError(padding_mode)
    if align_corners is None:
        align_corners = False
    return torch.ops.plenoxels.grid_sample_4d(
        input,
        grid,
        interpolation_mode,
        padding_mode_,
        align_corners,
    )
