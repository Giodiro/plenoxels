import itertools
import math
from typing import Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_grid import BaseGrid
from ...ops.interpolation import grid_sample_4d


class LearnableHashGrid(BaseGrid):
    def __init__(self,
                 grid_config: Union[str, List[List[Dict]]],
                 num_scenes: int = 1,
                 multiscale_type: str = 'cat',
                 **kwargs):
        if isinstance(grid_config, str):
            self.config: List[List[Dict]] = eval(grid_config)
        else:
            self.config: List[List[Dict]] = grid_config
        self.num_lods = len(self.config)
        super().__init__(self.num_lods)

        self.multiscale_type = multiscale_type
        self.scene_lod_grids = nn.ModuleList()
        self.lod_features = nn.ParameterList()
        feature_dim_list = []
        for si in range(num_scenes):
            lod_grids = nn.ModuleList()
            for lodi, lod_grid_config in enumerate(self.config):
                grids = nn.ParameterList()
                for li, grid_config in enumerate(lod_grid_config):
                    if "feature_dim" in grid_config and si == 0:
                        in_dim = grid_config["input_coordinate_dim"]
                        reso = grid_config["resolution"]
                        self.lod_features.append(nn.Parameter(nn.init.normal_(
                                torch.empty([grid_config["feature_dim"]] + [reso] * in_dim),
                                mean=0.0, std=grid_config["init_std"])))
                        feature_dim_list.append(grid_config["feature_dim"])
                    else:
                        in_dim = grid_config["input_coordinate_dim"]
                        out_dim = grid_config["output_coordinate_dim"]
                        grid_nd = grid_config["grid_dimensions"]
                        reso = grid_config["resolution"]
                        rank = grid_config["rank"]
                        num_comp = math.comb(in_dim, grid_nd)
                        # Configuration correctness checks
                        if li == 0:
                            assert in_dim == 3
                        assert out_dim in {2, 3, 4}
                        assert grid_nd <= in_dim
                        if grid_nd == in_dim:
                            assert rank == 1
                        grids.append(
                            nn.Parameter(nn.init.normal_(
                                torch.empty([num_comp, out_dim * rank] + [reso] * grid_nd),
                                mean=0.0, std=grid_config["init_std"]))
                        )
                lod_grids.append(grids)
            self.scene_lod_grids.append(lod_grids)
        assert len(self.lod_features) == self.num_lods
        # feature-dim must be the same in all LODs
        for fd in feature_dim_list:
            assert fd == feature_dim_list[0]
        self.feature_dim_ = feature_dim_list[0]

    def feature_dim(self, lod_idx=0):
        if self.multiscale_type == 'cat':
            return self.feature_dim_ * (lod_idx + 1)
        else:
            return self.feature_dim_

    def query_mask(self, coords, lod_idx, scene_idx=0):
        # TODO: Fix hardcoded
        return ((-1 <= coords) & (coords <= 1)).all(dim=-1)

    @staticmethod
    def get_coo_plane(coords, dim):
        """
        :param coords:
            torch tensor [n, input d]
        :param dim:
        :return:
            torch tensor [num_comp, n, dim]
        """
        coo_combs = list(itertools.combinations(range(coords.shape[-1]), dim))
        return coords[..., coo_combs].transpose(0, 1)

    @staticmethod
    def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        grid_dim = coords.shape[-1]

        if grid.dim() == grid_dim + 1:
            # no batch dimension present, need to add it
            grid = grid.unsqueeze(0)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        if grid_dim == 2:
            interp = F.grid_sample(
                grid,  # [B, feature_dim, reso, reso]
                coords[:, None, ...],  # [B, 1, n, 2]
                align_corners=True,
                mode='bilinear', padding_mode='border').squeeze().transpose(-1, -2)  # [B?, n, feature_dim]
        elif grid_dim == 3:
            interp = F.grid_sample(
                grid,  # [B, feature_dim, reso, reso, reso]
                coords[:, None, None, ...],  # [B, 1, 1, n, 3]
                align_corners=True,
                mode='bilinear', padding_mode='border').squeeze().transpose(-1, -2)  # [B?, n, feature_dim]
        elif grid_dim == 4:
            interp = grid_sample_4d(
                grid,  # [B, feature_dim, reso, reso, reso, reso]
                coords[:, None, None, None, ...],  # [B, 1, 1, 1, n, 4]
                align_corners=True,
                mode='bilinear', padding_mode='border').squeeze().transpose(-1, -2)  # [B?, n, feature_dim]
        else:
            raise ValueError("grid_dim can be 2, 3 or 4.")
        return interp

    def interpolate(self, coords, lod_idx, scene_idx=0) -> torch.Tensor:
        """
        :param coords:
            torch tensor [n, 3] must be in the range [-1, 1]
        :param lod_idx:
        :return:
            torch tensor [n, feature_dim]
        """
        lod_grids = self.scene_lod_grids[scene_idx]
        lod_grids_info = self.config
        lod_features = self.lod_features

        feats = []
        for lodi in range(lod_idx + 1):
            interp = coords
            for level_info, grid in zip(lod_grids_info[lodi], lod_grids[lodi]):
                if "feature_dim" in level_info:
                    continue
                coo_plane = self.get_coo_plane(
                    interp,
                    level_info.get("grid_dimensions", level_info["input_coordinate_dim"])
                )
                interp = self.grid_sample_wrapper(grid, coords).view(
                    grid.shape[0], -1, level_info["output_coordinate_dim"], level_info["rank"])
                interp = interp.prod(dim=0).sum(dim=-1)
            feats.append(self.grid_sample_wrapper(lod_features[lodi], interp))

        if lod_idx == 0:
            feats = feats[0]
        else:
            if self.multiscale_type == 'cat':
                feats = torch.cat(feats, dim=-1)
            elif self.multiscale_type == 'sum':
                feats = torch.stack(feats, dim=-1).sum(dim=-1)
            else:
                raise ValueError(f"{self.multiscale_type=}")
        return feats
