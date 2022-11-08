import logging as log
from abc import ABC
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


class LowrankModel(ABC, nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 sh: bool,
                 use_F: bool,
                 use_proposal_sampling: bool,
                 aabb: Union[List[torch.Tensor], torch.Tensor],
                 multiscale_res: Sequence[int],
                 num_scenes: int = 1,
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 density_activation: Optional[str] = 'trunc_exp',
                 density_model: Optional[str] = None,
                 # ray-sampling arguments
                 density_field_resolution: Optional[Sequence[int]] = None,
                 density_field_rank: Optional[int] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 n_intersections: Optional[int] = None,
                 single_jitter: bool = False,
                 raymarch_type: str = 'fixed',
                 spacing_fn: Optional[str] = None,
                 num_samples_multiplier: Optional[int] = None,
                 proposal_feature_dim: Optional[int] = None,
                 proposal_decoder_type: Optional[str] = None,
                 ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.multiscale_res = multiscale_res
        self.set_aabb(aabb)  # set_aabb handles both single tensor and a list.
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.density_model = density_model if density_model is not None else 'triplane'
        if self.density_model not in {'hexplane', 'triplane'}:
            log.warning(f'density model {self.density_model} is not recognized. '
                        f'Using triplane as default; other choice is hexplane.')
            self.density_model = 'triplane'
        self.sh = sh
        self.use_F = use_F
        self.use_proposal_sampling = use_proposal_sampling
        self.density_act = init_density_activation(density_activation)
        self.num_scenes = num_scenes
        self.timer = CudaTimer(enabled=False)
        self.proposal_feature_dim = proposal_feature_dim
        self.proposal_decoder_type = proposal_decoder_type

        self.pt_min, self.pt_max = None, None
        if self.use_F:
            self.pt_min = nn.ParameterList([
                torch.nn.Parameter(torch.tensor(-1.0)) for _ in range(len(self.multiscale_res))])
            self.pt_max = nn.ParameterList([
                torch.nn.Parameter(torch.tensor(+1.0)) for _ in range(len(self.multiscale_res))])

        self.spatial_distortion = None
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

    def step_cb(self, step, max_step):
        pass

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
            assert self.proposal_feature_dim is not None
            assert self.proposal_decoder_type is not None
            for reso in density_field_resolution:
                real_resolution = [reso] * 3
                if self.density_model == 'hexplane':
                    real_resolution.append(self.config[0]['resolution'][-1])
                field = TriplaneDensityField(
                    aabb=self.aabb(0),
                    resolution=real_resolution,
                    num_input_coords=4 if self.density_model == 'hexplane' else 3,
                    rank=density_field_rank,
                    spatial_distortion=self.spatial_distortion,
                    density_act=self.density_act,
                    len_time=self.len_time if self.density_model == 'hexplane' else None,
                    num_output_coords=self.proposal_feature_dim,
                    decoder_type=self.proposal_decoder_type,
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
