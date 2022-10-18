import itertools
import logging as log
import math
from typing import Dict, List, Union, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from plenoxels.models.utils import grid_sample_wrapper
from .decoders import NNDecoder, SHDecoder
from .lowrank_model import LowrankModel
from ..ops.activations import trunc_exp


class DensityMask(nn.Module):
    def __init__(self, density_volume: torch.Tensor, aabb: torch.Tensor):
        super().__init__()
        self.register_buffer('density_volume', density_volume)
        self.register_buffer('aabb', aabb)

    def sample_density(self, pts: torch.Tensor) -> torch.Tensor:
        # Normalize pts
        pts = (pts - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1
        pts = pts.to(dtype=self.density_volume.dtype)
        density_vals = grid_sample_wrapper(self.density_volume[None, ...], pts, align_corners=True).view(-1)
        return density_vals

    @property
    def grid_size(self) -> torch.Tensor:
        return torch.tensor((self.density_volume.shape[-1], self.density_volume.shape[-2], self.density_volume.shape[-3]), dtype=torch.long)


class LowrankLearnableHash(LowrankModel):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 aabb: List[torch.Tensor],
                 is_ndc: bool,
                 sh: bool,
                 render_n_samples: int,
                 num_scenes: int = 1,
                 **kwargs):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.set_aabb(aabb)
        self.extra_args = kwargs
        self.is_ndc = is_ndc
        self.sh = sh
        self.render_n_samples = render_n_samples
        self.density_multiplier = self.extra_args.get("density_multiplier")
        self.transfer_learning = self.extra_args["transfer_learning"]
        self.alpha_mask_threshold = self.extra_args["density_threshold"]

        self.density_act = lambda x: trunc_exp(x - 1)
        self.pt_min = torch.nn.Parameter(torch.tensor(-1.0))
        self.pt_max = torch.nn.Parameter(torch.tensor(1.0))

        self.scene_grids = nn.ModuleList()
        for si in range(num_scenes):
            grids = nn.ModuleList()
            for li, grid_config in enumerate(self.config):
                if "feature_dim" in grid_config and si == 0:  # Only make one set of features
                    self.features = self.init_features_param(grid_config, self.sh)
                    self.feature_dim = self.features.shape[0]
                else:
                    gpdesc = self.init_grid_param(grid_config, is_video=False, grid_level=li)
                    if li == 0:
                        self.set_resolution(gpdesc.reso, grid_id=si)
                    grids.append(gpdesc.grid_coefs)
            self.scene_grids.append(grids)
        if self.sh:
            self.decoder = SHDecoder(feature_dim=self.feature_dim)
        else:
            self.decoder = NNDecoder(feature_dim=self.feature_dim, sigma_net_width=64, sigma_net_layers=1)
        log.info(f"Initialized LearnableHashGrid with {num_scenes} scenes, decoder: {self.decoder}.")

    def compute_features(self,
                         pts: torch.Tensor,
                         grid_id: int,
                         return_coords: bool = False
                         ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param pts:
            Coordinates normalized between -1, 1
        :param grid_id:
        :param return_coords:
        :return:
        """
        grids: nn.ModuleList = self.scene_grids[grid_id]  # noqa
        grids_info = self.config

        interp = pts
        grid: nn.ParameterList
        for level_info, grid in zip(grids_info, grids):
            if "feature_dim" in level_info:
                continue
            coo_combs = list(itertools.combinations(
                range(interp.shape[-1]),
                level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))
            interp_out = None
            for ci, coo_comb in enumerate(coo_combs):
                if interp_out is None:
                    interp_out = (
                        grid_sample_wrapper(grid[ci], interp[..., coo_comb]).view(
                            -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
                else:
                    interp_out = interp_out * (
                        grid_sample_wrapper(grid[ci], interp[..., coo_comb]).view(
                            -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
            interp = interp_out.mean(dim=-1)

        if interp.numel() > 0:
            interp = (interp - self.pt_min) / (self.pt_max - self.pt_min)
            interp = interp * 2 - 1

        out = grid_sample_wrapper(self.features.to(dtype=interp.dtype), interp).view(-1, self.feature_dim)
        if return_coords:
            return out, interp
        return out

    @torch.no_grad()
    def shrink(self, occ_grid, grid_id: int):
        log.info(f"Calculating occupancy {grid_id}")
        aabb = self.aabb(grid_id)
        occ_grid_reso = occ_grid.resolution
        pts = occ_grid.grid_coords.view([*occ_grid_reso, 3])
        # Transpose to get correct Depth, Height, Width format
        pts = pts.transpose(0, 2).contiguous()
        mask = occ_grid.binary.transpose(0, 2).contiguous()
        valid_pts = pts[mask]
        pts_min = valid_pts.amin(0)
        pts_max = valid_pts.amax(0)
        # Normalize pts_min, pts_max from [0, reso] to world-coordinates
        pts_min = (pts_min / occ_grid_reso)
        pts_min = aabb[0] * (1 - pts_min) + aabb[1] * pts_min
        pts_max = (pts_max / occ_grid_reso)
        pts_max = aabb[0] * (1 - pts_max) + aabb[1] * pts_max
        new_aabb = torch.stack((pts_min, pts_max), 0)
        log.info(f"Scene {grid_id} can be shrunk to new bounding box: {new_aabb.view(-1)}.")

        log.info(f"Shrinking grid {grid_id}...")
        cur_grid_size = self.resolution(grid_id)

        cur_units = (aabb[1] - aabb[0]) / (cur_grid_size - 1)
        t_l, b_r = (new_aabb[0] - aabb[0]) / cur_units, (new_aabb[1] - aabb[0]) / cur_units
        t_l = torch.round(t_l).long()
        b_r = torch.round(b_r).long() + 1
        b_r = torch.minimum(b_r, cur_grid_size)  # don't exceed current grid dimensions

        # Truncate the parameter grid to the new grid-size
        # IMPORTANT: This will only work if input-dim is 3!
        grid_info = self.config[0]
        coo_combs = list(itertools.combinations(
            range(grid_info["input_coordinate_dim"]),
            grid_info.get("grid_dimensions", grid_info["input_coordinate_dim"])))
        for ci, coo_comb in enumerate(coo_combs):
            slices = [slice(None), slice(None)] + [slice(t_l[cc].item(), b_r[cc].item()) for cc in coo_comb[::-1]]
            self.scene_grids[grid_id][0][ci] = torch.nn.Parameter(
                self.scene_grids[grid_id][0][ci].data[slices]
            )

        # TODO: Why the correction? Check if this ever occurs
        if not torch.all(occ_grid_reso == cur_grid_size):
             t_l_r, b_r_r = t_l / (cur_grid_size - 1), (b_r - 1) / (cur_grid_size - 1)
             correct_aabb = torch.zeros_like(new_aabb)
             correct_aabb[0] = (1 - t_l_r) * aabb[0] + t_l_r * aabb[1]
             correct_aabb[1] = (1 - b_r_r) * aabb[0] + b_r_r * aabb[1]
             log.info(f"Corrected new AABB from {new_aabb.view(-1)} to {correct_aabb.view(-1)}")
             new_aabb = correct_aabb

        new_size = b_r - t_l
        self.set_aabb(new_aabb, grid_id)
        self.set_resolution(new_size, grid_id)
        log.info(f"Shrunk scene {grid_id}. New AABB={new_aabb.view(-1)} New resolution={new_size.view(-1)}")

    @torch.no_grad()
    def upsample(self, new_reso, grid_id: int):
        grid_info = self.config[0]
        coo_combs = list(itertools.combinations(
            range(grid_info["input_coordinate_dim"]),
            grid_info.get("grid_dimensions", grid_info["input_coordinate_dim"])))
        for ci, coo_comb in enumerate(coo_combs):
            new_size = [new_reso[cc] for cc in coo_comb]
            if len(coo_comb) == 3:
                mode = 'trilinear'
            elif len(coo_comb) == 2:
                mode = 'bilinear'
            elif len(coo_comb) == 1:
                mode = 'linear'
            else:
                raise RuntimeError()
            grid_data = self.scene_grids[grid_id][0][ci].data
            self.scene_grids[grid_id][0][ci] = torch.nn.Parameter(
                F.interpolate(grid_data, size=new_size[::-1], mode=mode, align_corners=True))
        self.set_resolution(
            torch.tensor(new_reso, dtype=torch.long, device=grid_data.device), grid_id)
        log.info(f"Upsampled scene {grid_id} to resolution={new_reso}")

    def step_size(self, n_samples: int, grid_id: int):
        aabb = self.aabb(grid_id)
        return (
            (aabb[1] - aabb[0]).max()
            * math.sqrt(3)
            / n_samples
        )

    def query_opacity(self, pts: torch.Tensor, grid_id: int):
        density = self.query_density(pts, grid_id)
        opacity = density * self.step_size(self.render_n_samples, grid_id)
        return opacity

    def query_density(self, pts: torch.Tensor, grid_id: int, return_feat: bool = False):
        pts_norm = self.normalize_coords(pts, grid_id)
        selector = ((pts_norm >= -1.0) & (pts_norm <= 1.0)).all(dim=-1)

        features = self.compute_features(pts_norm, grid_id)
        density = (
            self.density_act(self.decoder.compute_density(
                features, rays_d=None, precompute_color=False)).view((*pts_norm.shape[:-1], 1))
            * selector[..., None]
        )
        if return_feat:
            return density, features
        return density

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        grid_id: int,
    ):
        density, embedding = self.query_density(rays_o, grid_id, return_feat=True)
        rgb = self.decoder.compute_color(embedding, rays_d=rays_d)
        return rgb, density

    def get_params(self, lr):
        params = [
            {"params": self.scene_grids.parameters(), "lr": lr},
            {"params": [self.pt_min, self.pt_max], "lr": lr},
            #{"params": self.bn.parameters(), "lr": lr},
        ]
        if not self.transfer_learning:
            params += [
                {"params": self.decoder.parameters(), "lr": lr},
                {"params": self.features, "lr": lr},
            ]
        return params
