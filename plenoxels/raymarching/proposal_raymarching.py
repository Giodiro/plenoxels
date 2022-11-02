import math
from typing import Optional, Mapping
import logging as log
import numpy as np
import torch
from plenoxels.models.utils import raw2alpha, init_density_activation
# For now, only implemented assuming scene contraction.
# To use for another type of scene, just change the initial sampler and the mapping from ray samples to model samples
# Based on Nerfactory https://github.com/nerfstudio-project/nerfstudio/blob/be70b86b181457fbf98329c74721f092b7715e2d/nerfstudio/model_components/ray_samplers.py#L499
# Which is in turn based on MipNerf360

# TODO: remember to call the loss function and optimize this density model
# Based on https://github.com/nerfstudio-project/nerfstudio/blob/be70b86b181457fbf98329c74721f092b7715e2d/nerfstudio/model_components/losses.py#L93
def histogram_loss(fine_samples, fine_weights, initial_samples, initial_weights):
    # TODO

# Based on https://github.com/nerfstudio-project/nerfstudio/blob/be70b86b181457fbf98329c74721f092b7715e2d/nerfstudio/model_components/ray_samplers.py#L249
class PDFSampler():
    # TODO

class TriplaneDensity():
    def __init__(self, resolution, rank):
        self.grids = 
    # TODO

# Two-stage raymarching using a proposal model, following MipNerf360
class ProposalSampler():
    def __init__(self,
                 n_initial_samples: int = 128,
                 n_fine_samples: int = 128,
                 resolution: List[int] = [64, 64, 64],
                 rank: int = 20):
        self.initial_sampler = LinearThenReciprocalSampler(num_samples=n_initial_samples)
        self.pdf_sampler = PDFSampler(num_samples=n_fine_samples)
        self.density_model = TriplaneDensity(resolution=resolution, rank=rank)
        self.density_act = init_density_activation(
            self.extra_args.get('density_activation', 'trunc_exp'))

    def get_samples(self, rays_o, rays_d, near, far, global_scale=torch.tensor([0, 0, 0]), global_translation=torch.tensor([1, 1, 1])):
        # Get initial samples in ray space
        initial_samples, deltas = self.initial_sampler.generate_samples(near, far)  # [n_rays, n_initial_samples]
        # Map them to model space
        world_samples = ray_to_model(rays_o, rays_d, initial_samples, global_scale, global_translation)  # [n_rays, n_initial_samples, 3]
        # Get the density at each initial sample
        sigmas = self.density_model(world_samples)  # [n_rays, n_initial_samples]
        # Get the weights corresponding to these densities
        _, initial_weights, _ = raw2alpha(self.density_act(sigmas), deltas)  # each is shape [n_rays, n_initial_samples]
        # Use these weights to call the pdf sampler
        fine_samples, deltas = self.pdf_sampler(near, far, initial_samples, initial_weights)
        # Map them to model space
        world_samples = ray_to_model(rays_o, rays_d, fine_samples, global_scale, global_translation)  # [n_rays, n_initial_samples, 3]
        # world_samples and deltas are used to query the full model. 
        # fine_samples, initial_samples, and initial_weights are used along with full model weights for world_samples, to compute histogram_loss and update density_model
        return world_samples, deltas, fine_samples, initial_samples, initial_weights


class LinearThenReciprocalSampler():
    def __init__(self, num_samples):
        self.num = num_samples
    
    # Generate samples in ray space
    def generate_samples(near, far):
        # Put half the samples linearly spaced up to a distance of 2, then linear in disparity after that up to the far bound
        x1 = genspace(near, near + 2, self.num // 2, lambda x: x, lambda x: x)
        x2 = genspace(near + 2, far, self.num // 2, torch.log, torch.exp)
        samples = torch.cat([x1, x2], dim=-1)
        deltas = intersections.diff(dim=-1)
        return samples[:, :-1], deltas


# Map samples from ray-space [n_rays, n_samples] to model-space [n_rays, n_samples, 3]
def ray_to_model(rays_o, rays_d, ray_samples, global_scale, global_translation):
    # Normalize rays_d
    dir_norm = torch.linalg.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d / dir_norm
    # Map from ray space to world space
    samples = rays_o[..., None, :] + rays_d[..., None, :] * ray_samples[..., None]  # [n_rays, n_samples, 3]
    # Apply global translation and scale
    samples = samples * global_scale[None, None, :].to(samples.device) + global_translation[None, None, :].to(samples.device)
    # Apply scene contraction
    samples = contract(samples)
    return samples


def contract(pts):
    """
    Apply the square (L_infinity norm) version of the scene contraction from MipNeRF360

    :param pts:
        torch.Tensor of shape [n_rays, n_samples, 3]
    :return:
        the contracted tensor with the same shape as `pts`
    """
    norms = torch.linalg.vector_norm(
        pts, ord=np.inf, dim=-1, keepdim=True).expand(-1, -1, 3)  # [n_rays, n_samples, 3]
    norm_mask = norms > 1
    pts[norm_mask] = (2.0 - 1.0 / norms[norm_mask]) * pts[norm_mask] / norms[norm_mask]
    return pts