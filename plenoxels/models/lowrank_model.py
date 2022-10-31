import collections.abc
import itertools
import math
from abc import ABC
from dataclasses import dataclass
from typing import List, Sequence, Optional, Union

import torch
import torch.nn as nn


@dataclass
class GridParamDescription:
    grid_coefs: nn.ParameterList
    reso: torch.Tensor
    time_reso: int = None
    time_coef: nn.Parameter = None


class LowrankModel(ABC, nn.Module):
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
            torch.empty([grid_config["feature_dim"]] + reso[::-1]))
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
                grid_coefs.append(
                    nn.Parameter(nn.init.uniform_(torch.empty(
                        [1, out_dim * rank[ci]] + [reso[cc] for cc in coo_comb[::-1]]
                    ), a=0.1, b=0.5)))  
        
        if is_appearance:  
            time_reso = int(grid_config["time_reso"])
            if use_F:
                time_coef = nn.Parameter(nn.init.uniform_(
                    torch.empty([out_dim * rank[0], time_reso]),
                    a=-1.0, b=1.0))  # if time init is fixed at 1, then it learns a static video
            else:
                time_coef = nn.Parameter(nn.init.ones_(torch.empty([out_dim * rank[0], time_reso])))  # no time dependence
            return GridParamDescription(
                grid_coefs=grid_coefs, reso=pt_reso, time_reso=time_reso, time_coef=time_coef)
        return GridParamDescription(
            grid_coefs=grid_coefs, reso=pt_reso)



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