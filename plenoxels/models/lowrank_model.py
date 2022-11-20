import collections.abc
import itertools
import math
from abc import ABC
from dataclasses import dataclass
from typing import List, Sequence, Optional, Union, Dict

import torch
import torch.nn as nn

from plenoxels.models.decoders import SHDecoder, NNDecoder, BaseDecoder
from plenoxels.models.utils import init_density_activation, grid_sample_wrapper


@dataclass
class GridParamDescription:
    grid_coefs: nn.ParameterList
    reso: torch.Tensor
    time_reso: int = None
    time_coef: nn.Parameter = None


class LowrankModel(ABC, nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 sh: bool,
                 use_F: bool,
                 density_activation: str,
                 aabb: Union[Sequence[torch.Tensor], torch.Tensor]):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.sh = sh
        self.use_F = use_F
        self.density_act = init_density_activation(density_activation)
        self.set_aabb(aabb)

    def set_aabb(self, aabb: Union[torch.Tensor, List[torch.Tensor]], grid_id: Optional[int] = None):
        if grid_id is None:
            if isinstance(aabb, list):
                # aabb needs to be BufferList (but BufferList doesn't exist so we emulate it)
                for i, p in enumerate(aabb):
                    assert p.shape == (2, 3)
                    if hasattr(self, f'aabb{i}'):
                        setattr(self, f'aabb{i}', p)
                    else:
                        self.register_buffer(f'aabb{i}', p)
            else:
                assert isinstance(aabb, torch.Tensor)
                assert aabb.shape == (2, 3)
                self.register_buffer('aabb0', aabb)
        else:
            assert isinstance(aabb, torch.Tensor)
            assert aabb.shape == (2, 3)
            if hasattr(self, f'aabb{grid_id}'):
                setattr(self, f'aabb{grid_id}', aabb)
            else:
                self.register_buffer(f'aabb{grid_id}', aabb)

    def aabb(self, i: int = 0) -> torch.Tensor:
        return getattr(self, f'aabb{i}')

    def set_resolution(self, resolution: Union[torch.Tensor, List[torch.Tensor]], grid_id: Optional[int] = None):
        if grid_id is None:
            # resolution needs to be BufferList (but BufferList doesn't exist so we emulate it)
            for i, p in enumerate(resolution):
                if hasattr(self, f'resolution{i}'):
                    setattr(self, f'resolution{i}', p)
                else:
                    self.register_buffer(f'resolution{i}', p)
        else:
            assert isinstance(resolution, torch.Tensor)
            if hasattr(self, f'resolution{grid_id}'):
                setattr(self, f'resolution{grid_id}', resolution)
            else:
                self.register_buffer(f'resolution{grid_id}', resolution)

    def resolution(self, i: int) -> torch.Tensor:
        return getattr(self, f'resolution{i}')

    def init_decoder(self) -> BaseDecoder:
        if self.sh:
            return SHDecoder(feature_dim=self.feature_dim)
        else:
            return NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)

    @torch.autograd.no_grad()
    def normalize_coords(self, pts: torch.Tensor, grid_id: int = 0) -> torch.Tensor:
        """
        break-down of the normalization steps. pts starts from [a0, a1]
        1. pts - a0 => [0, a1-a0]
        2. / (a1 - a0) => [0, 1]
        3. * 2 => [0, 2]
        4. - 1 => [-1, 1]
        """
        aabb = self.aabb(grid_id)
        return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1

    @staticmethod
    def interpolate_ms_features(pts: torch.Tensor,
                                ms_grids: nn.ModuleList,
                                grid_info: Dict[str, int],
                                feature_dim: int) -> torch.Tensor:
        coo_combs = list(itertools.combinations(
            range(pts.shape[-1]),
            grid_info.get("grid_dimensions", grid_info["input_coordinate_dim"]))
        )
        multi_scale_interp = torch.zeros_like(pts[0, 0])
        grid: nn.ParameterList
        for scale_id, grid in enumerate(ms_grids):
            interp_space = torch.ones_like(pts[0, 0])  # [n, F_dim]
            for ci, coo_comb in enumerate(coo_combs):
                # interpolate in plane
                interp_out_plane = (
                    grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                    .view(-1, feature_dim)
                )
                # compute product
                interp_space = interp_space * interp_out_plane
            # sum over scales
            multi_scale_interp = multi_scale_interp + interp_space
        return multi_scale_interp

    @staticmethod
    def init_features_param(grid_config, sh: bool) -> torch.nn.Parameter:
        assert "feature_dim" in grid_config

        reso: List[int] = grid_config["resolution"]
        try:
            in_dim = len(reso)
        except AttributeError:
            raise ValueError("Configuration incorrect: resolution must be a list.")
        assert in_dim == grid_config["input_coordinate_dim"]
        features = nn.Parameter(
            torch.empty([grid_config["feature_dim"]] + reso[::-1]))
        if sh:
            nn.init.zeros_(features)
            features[-1].data.fill_(grid_config["init_std"])  # here init_std is repurposed as the sigma initialization
        else:
            nn.init.normal_(features, mean=0.0, std=grid_config["init_std"])
        return features

    @staticmethod
    def init_grid_param(grid_config, is_video: bool, grid_level: int, use_F: bool) -> GridParamDescription:
        out_dim: int = grid_config["output_coordinate_dim"]
        grid_nd: int = grid_config["grid_dimensions"]
        reso: List[int] = grid_config["resolution"]
        try:
            in_dim = len(reso)
        except AttributeError:
            raise ValueError("Configuration incorrect: resolution must be a list.")
        pt_reso = torch.tensor(reso, dtype=torch.long)
        num_comp = math.comb(in_dim, grid_nd)
        # Configuration correctness checks
        assert in_dim == grid_config["input_coordinate_dim"]
        if grid_level == 0:
            if is_video:
                assert in_dim in {3, 4}
            else:
                assert in_dim == 3
        if use_F:
            assert out_dim in {1, 2, 3, 4, 5, 6, 7}
        assert grid_nd <= in_dim
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        grid_coefs = nn.ParameterList()
        for ci, coo_comb in enumerate(coo_combs):
            if use_F:  # TODO: not updated
                grid_coefs.append(
                    nn.Parameter(nn.init.uniform_(torch.empty(
                        [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
                    ), a=-1.0, b=1.0)))
            else:
                grid_coefs.append(
                    nn.Parameter(torch.empty(
                        [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
                    )))
                if is_video and 3 in coo_comb:  # is a time-plane
                    nn.init.ones_(grid_coefs[-1])
                else:
                    nn.init.uniform_(grid_coefs[-1], a=0.1, b=0.5)
        return GridParamDescription(
            grid_coefs=grid_coefs, reso=pt_reso)


def to_list(el, list_len, name: Optional[str] = None) -> Sequence:
    if not isinstance(el, collections.abc.Sequence):
        return [el] * list_len
    if len(el) != list_len:
        raise ValueError(f"Length of {name} is incorrect. Expected {list_len} but found {len(el)}")
    return el
