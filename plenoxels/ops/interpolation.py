from importlib.machinery import PathFinder
from pathlib import Path
from typing import Optional

import torch

try:
    spec = PathFinder().find_spec("c_ext", [str(Path(__file__).resolve().parents[1])])
    torch.ops.load_library(spec.origin)
except Exception:
    print("Couldn't load c_ext")


def get_interpolation_mode(mode: str) -> int:
    if mode == "bilinear":
        return 0
    elif mode == "nearest":
        return 1
    elif mode == "bicubic":
        return 2
    else:
        raise ValueError(mode)


def get_padding_mode(mode: str) -> int:
    if mode == "zeros":
        return 0
    elif mode == "border":
        return 1
    elif mode == "reflection":
        return 2
    else:
        raise ValueError(mode)


def grid_sample_1d(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: Optional[bool] = None,
) -> torch.Tensor:
    interpolation_mode = get_interpolation_mode(mode)
    padding_mode_ = get_padding_mode(padding_mode)
    if align_corners is None:
        align_corners = False
    return torch.ops.plenoxels.grid_sample_1d(
        input,
        grid,
        interpolation_mode,
        padding_mode_,
        align_corners,
    )


def grid_sample_nd(
        input: torch.Tensor,
        grid: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: Optional[bool] = None,
) -> torch.Tensor:
    interpolation_mode = get_interpolation_mode(mode)
    padding_mode_ = get_padding_mode(padding_mode)
    if align_corners is None:
        align_corners = False
    return torch.ops.plenoxels.grid_sample_nd(
        input,
        grid,
        interpolation_mode,
        padding_mode_,
        align_corners,
    )


def grid_sample_4d(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    interpolation_mode = get_interpolation_mode(mode)
    padding_mode_ = get_padding_mode(padding_mode)
    if align_corners is None:
        align_corners = False
    return torch.ops.plenoxels.grid_sample_4d(
        input,
        grid,
        interpolation_mode,
        padding_mode_,
        align_corners,
    )
