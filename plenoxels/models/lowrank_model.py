import collections.abc
import itertools
import math
from abc import ABC
from dataclasses import dataclass
import logging as log
from typing import List, Sequence, Optional, Union, Dict, Tuple, Callable, Any

import torch
import torch.nn as nn

from plenoxels.models.density_fields import TriplaneDensityField
from plenoxels.models.utils import init_density_activation
from plenoxels.raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler
)
from plenoxels.raymarching.raymarching import RayMarcher
from plenoxels.raymarching.spatial_distortions import SceneContraction
from plenoxels.runners.timer import CudaTimer


@dataclass
class GridParamDescription:
    grid_coefs: nn.ParameterList
    reso: torch.Tensor
    time_reso: int = None
    time_coef: nn.Parameter = None


class LowrankModel(ABC, nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 sh: bool,
                 use_F: bool,
                 use_proposal_sampling: bool,

                 num_scenes: int = 1,
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 density_activation: Optional[str] = 'trunc_exp',
                 # ray-sampling arguments
                 density_field_resolution: Optional[Sequence[int]] = None,
                 density_field_rank: Optional[int] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 n_intersections: Optional[int] = None,
                 single_jitter: bool = False,
                 raymarch_type: str = 'fixed',
                 spacing_fn: Optional[str] = None,
                 num_samples_multiplier: Optional[int] = None,
                 ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.is_video = self.config[0]['input_coordinate_dim'] == 4
        self.sh = sh
        self.use_F = use_F
        self.use_proposal_sampling = use_proposal_sampling
        self.density_act = init_density_activation(density_activation)
        self.num_scenes = num_scenes
        self.timer = CudaTimer(enabled=False)

        self.pt_min, self.pt_max = None, None
        if self.use_F:
            self.pt_min = torch.nn.Parameter(torch.tensor(-1.0))
            self.pt_max = torch.nn.Parameter(torch.tensor(1.0))

        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'),
                global_scale=global_scale,
                global_translation=global_translation)

        self.raymarcher, self.density_fields, self.density_fns = self.init_raymarcher(
            prop_sampling=self.use_proposal_sampling,
            density_field_resolution=density_field_resolution,
            density_field_rank=density_field_rank,
            num_proposal_samples=num_proposal_samples,
            n_intersections=n_intersections,
            single_jitter=single_jitter,
            raymarch_type=raymarch_type,
            spacing_fn=spacing_fn,
            num_sample_multiplier=num_samples_multiplier,
        )

    def set_aabb(self, aabb: Union[torch.Tensor, List[torch.Tensor]], grid_id: Optional[int] = None):
        if grid_id is None:
            # aabb needs to be BufferList (but BufferList doesn't exist so we emulate it)
            for i, p in enumerate(aabb):
                if hasattr(self, f'aabb{i}'):
                    setattr(self, f'aabb{i}', p)
                else:
                    self.register_buffer(f'aabb{i}', p)
        else:
            assert isinstance(aabb, torch.Tensor)
            if hasattr(self, f'aabb{grid_id}'):
                setattr(self, f'aabb{grid_id}', aabb)
            else:
                self.register_buffer(f'aabb{grid_id}', aabb)

    def aabb(self, i: int) -> torch.Tensor:
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
    def init_features_param(grid_config, sh: bool) -> torch.nn.Parameter:
        assert "feature_dim" in grid_config

        reso: List[int] = grid_config["resolution"]
        try:
            in_dim = len(reso)
        except AttributeError:
            raise ValueError("Configuration incorrect: resolution must be a list.")
        assert in_dim == grid_config["input_coordinate_dim"]
        features = nn.Parameter(
            torch.zeros([grid_config["feature_dim"]] + reso[::-1]))
        if sh:
            if reso[0] > 2:
                nn.init.zeros_(features)
                features[-1].data.fill_(grid_config["init_std"])  # here init_std is repurposed as the sigma initialization
            elif reso[0] == 2:
                # Make each feature a standard basis vector
                # Feature shape is [feature_dim] + [2]*d
                nn.init.uniform_(features, a=0, b=0.1)
                feats = features.data.view(grid_config["feature_dim"], -1).permute(1, 0)  # [feature_dim, num_features]
                for i in range(grid_config["feature_dim"]-1):
                    feats[i] = basis_vector(grid_config["feature_dim"], i, dense=False)
                # For trying a fixed/nonlearnable F
                # nn.init.uniform_(features, a=0, b=0)  # for learnable, works well to have a=0, b=0.1
                # feats = features.data.view(grid_config["feature_dim"], -1).permute(1, 0)  # [feature_dim, num_features]
                # for i in range(grid_config["feature_dim"]-1):
                #     feats[i] = basis_vector(grid_config["feature_dim"], i, dense=True)
                # extra_sigma_vals = [-100, 100, 1000, -1000]
                # k = 0
                # for j in range(grid_config["feature_dim"], len(feats)):
                #     feats[j] = basis_vector(grid_config["feature_dim"], i) * extra_sigma_vals[k]
                #     k = k + 1
                # feats[grid_config["feature_dim"]]
                print(feats)
                features.data = feats.permute(0, 1).reshape([grid_config["feature_dim"]] + reso[::-1])
        else:
            nn.init.normal_(features, mean=0.0, std=grid_config["init_std"])
        return features

    @staticmethod
    def init_grid_param(grid_config, is_video: bool, is_appearance: bool, grid_level: int, use_F: bool = True) -> GridParamDescription:
        out_dim: int = grid_config["output_coordinate_dim"]
        grid_nd: int = grid_config["grid_dimensions"]
        
        reso: List[int] = grid_config["resolution"]
        try:
            in_dim = len(reso)
        except AttributeError:
            raise ValueError("Configuration incorrect: resolution must be a list.")
        pt_reso = torch.tensor(reso, dtype=torch.long)
        num_comp = math.comb(in_dim, grid_nd)
        rank: Sequence[int] = to_list(grid_config["rank"], num_comp, "rank")
        grid_config["rank"] = rank
        # Configuration correctness checks
        assert in_dim == grid_config["input_coordinate_dim"]
        if grid_level == 0:
            if is_video:
                assert in_dim == 4
            else:
                assert in_dim == 3 or in_dim == 4
        if use_F:
            assert out_dim in {1, 2, 3, 4, 5, 6, 7}
        assert grid_nd <= in_dim
        if grid_nd == in_dim:
            assert all(r == 1 for r in rank)
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
        grid_coefs = nn.ParameterList()
        for ci, coo_comb in enumerate(coo_combs):
            if use_F:
                # if appearance and time plane, then init as ones (static).
                if is_appearance and 3 in coo_comb:
                    grid_coefs.append(
                        nn.Parameter(nn.init.ones_(torch.empty(
                            [1, out_dim * rank[ci]] + [reso[cc] for cc in coo_comb[::-1]]
                        ))))
                else:
                    grid_coefs.append(
                        nn.Parameter(nn.init.uniform_(torch.empty(
                            [1, out_dim * rank[ci]] + [reso[cc] for cc in coo_comb[::-1]]
                        ), a=-1.0, b=1.0)))
            else:
                if is_appearance and 3 in coo_comb:
                    grid_coefs.append(
                        nn.Parameter(nn.init.ones_(torch.empty(
                            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
                        ))))
                else:
                    grid_coefs.append(
                        nn.Parameter(nn.init.uniform_(torch.empty(
                            [1, out_dim * rank[ci]] + [reso[cc] for cc in coo_comb[::-1]]
                        ), a=0.1, b=0.5)))
        """
        if is_appearance:
            time_reso = int(grid_config["time_reso"])

            if use_F:
                time_coef = nn.Parameter(nn.init.uniform_(
                    torch.empty([out_dim * rank[0], time_reso]),
                    a=-1.0, b=1.0))  # if time init is fixed at 1, then it learns a static video
            else:

                # if sh + density in grid, then we do not want appearance code to influence density
                if out_dim == 28:
                    out_dim = out_dim - 1

                time_coef = nn.Parameter(nn.init.ones_(torch.empty([out_dim * rank[0], time_reso])))  # no time dependence
            return GridParamDescription(
                grid_coefs=grid_coefs, reso=pt_reso, time_reso=time_reso, time_coef=time_coef)
        """
        return GridParamDescription(
            grid_coefs=grid_coefs, reso=pt_reso)

    def init_raymarcher(self,
                        prop_sampling: bool,
                        density_field_resolution: Optional[Sequence[int]],
                        density_field_rank: Optional[int],
                        single_jitter: bool,
                        num_proposal_samples: Optional[Tuple[int]],
                        n_intersections: int,
                        num_sample_multiplier: Optional[int],
                        raymarch_type: str,
                        spacing_fn: Optional[str],
                        ) -> Tuple[Any, torch.nn.ModuleList, List[Callable]]:
        density_fields, density_fns = torch.nn.ModuleList(), []
        if prop_sampling:
            if self.num_scenes != 1:
                raise NotImplementedError("Proposal sampling with multiple scenes not implemented.")
            assert num_proposal_samples is not None
            assert density_field_rank is not None
            assert density_field_resolution is not None
            for reso in density_field_resolution:
                real_resolution = [reso] * 3
                if self.is_video:
                    real_resolution.append(self.config[0]['resolution'][-1])
                field = TriplaneDensityField(
                    aabb=self.aabb(0),
                    resolution=real_resolution,
                    num_input_coords=4 if self.is_video else 3,
                    rank=density_field_rank,
                    spatial_distortion=self.spatial_distortion,
                    density_act=self.density_act,
                    len_time=self.len_time if self.is_video else None,
                )
                density_fields.append(field)
                density_fns.append(field.get_density)
            if raymarch_type != 'fixed':
                log.warning("raymarch_type is not 'fixed' but we will use 'n_intersections' anyways.")
            if self.is_contracted:
                initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
            else:
                initial_sampler = UniformSampler(single_jitter=single_jitter)
            raymarcher = ProposalNetworkSampler(
                num_proposal_samples_per_ray=num_proposal_samples,
                num_nerf_samples_per_ray=n_intersections,
                num_proposal_network_iterations=len(num_proposal_samples),
                single_jitter=single_jitter,
                initial_sampler=initial_sampler,
            )
        else:
            assert spacing_fn is not None
            assert num_sample_multiplier is not None
            raymarcher = RayMarcher(
                n_intersections=n_intersections,
                num_sample_multiplier=num_sample_multiplier,
                raymarch_type=raymarch_type,
                spacing_fn=spacing_fn,
                single_jitter=single_jitter,
                spatial_distortion=self.spatial_distortion)
        return raymarcher, density_fields, density_fns


def to_list(el, list_len, name: Optional[str] = None) -> Sequence:
    if not isinstance(el, collections.abc.Sequence):
        return [el] * list_len
    if len(el) != list_len:
        raise ValueError(f"Length of {name} is incorrect. Expected {list_len} but found {len(el)}")
    return el


def basis_vector(n, k, dense=True):
    vector = torch.zeros(n)
    vector[k] = 1
    if dense:
        vector[-1] = 10
    return vector
