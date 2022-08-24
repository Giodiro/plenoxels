import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


class SHRender(nn.Module):
    def __init__(self, sh_degree: int):
        super().__init__()
        self.sh_degree = sh_degree
        self.direction_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": sh_degree + 1,
            },
        )

    def compute_density(self, density_features, mask, rays_d):
        dim_batch, dim_nintrs = mask.shape
        sigma = torch.zeros(dim_batch, dim_nintrs, device=density_features.device)

        if mask.any():
            sigma_valid = F.relu(density_features[:, 0])
            sigma[mask] = sigma_valid
        return sigma

    def compute_color(self, color_features, mask, rays_d):
        dim_batch, dim_nintrs = mask.shape

        rgb = torch.zeros((dim_batch, dim_nintrs, 3), device=color_features.device)

        if mask.any():
            # 3. Create SH coefficients and mask them
            sh_mult = self.direction_encoder(rays_d).unsqueeze(1).expand(dim_batch, dim_nintrs, -1)  # [batch, nintrs, ch/3]
            sh_mult = sh_mult[mask].unsqueeze(1)  # [mask_pts, 1, ch/3]
            # 4. Interpolate rgbdata, use SH coefficients to get to RGB
            sh_masked = color_features[:, 1:].reshape(-1, 3, sh_mult.shape[-1])  # [mask_pts, 3, ch/3]
            rgb_masked = torch.sum(sh_mult * sh_masked, dim=-1)  # [mask_pts, 3]
            rgb[mask] = rgb_masked

        return rgb

    def __repr__(self):
        return f"SHRender(sh_degree={self.sh_degree})"
