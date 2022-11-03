import torch
import tinycudann as tcnn
import numpy as np

from .base_decoder import BaseDecoder


# From https://github.com/sxyu/svox2/blob/07f41aea51f88380d2e492fb4b7c7099afc1e81f/svox2/utils.py#L115
def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """
    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    SH_C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    SH_C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    SH_C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)
            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)
                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return result


class SHDecoder(BaseDecoder):
    def __init__(self, feature_dim: int, decoder_type: str = 'manual'):
        super().__init__()
        sh_degree = int(round(np.sqrt((feature_dim - 1) / 3) - 1))
        if feature_dim - 1 != ((sh_degree + 1) ** 2) * 3:
            raise ValueError(f"feature_dim is incorrect for SHDecoder with {sh_degree} degrees")
        print(f'using spherical harmonic degree {sh_degree}, which corresponds to {feature_dim} features')
        self.sh_degree = sh_degree + 1
        self.sh_dim = (feature_dim - 1) // 3
        if decoder_type not in {'manual', 'tcnn'}:
            raise ValueError(
                f"SH decoder type must be either 'manual' or 'tcnn', but found {decoder_type}.")
        self.decoder_type = decoder_type
        self.direction_encoder = None
        if self.decoder_type == 'tcnn':
            self.direction_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": self.sh_degree,
                },
            )

    def compute_density(self, features, rays_d, **kwargs):
        return features[..., -1].view(-1, 1)

    def compute_color(self, features, rays_d):
        color_features = features[..., :-1]
        if self.decoder_type == 'tcnn':
            rays_d = (rays_d + 1.0) / 2.0
            sh_mult = self.direction_encoder(rays_d)[:, None, :]  # [batch, 1, harmonic_components]
        else:
            sh_mult = eval_sh_bases(self.sh_dim, rays_d)[:, None, :]  # [batch, 1, harmonic_components]
        rgb = color_features.view(color_features.shape[0], 3, sh_mult.shape[-1])  # [batch, 3, harmonic_components]
        return torch.sum(sh_mult * rgb, dim=-1)  # [batch, 3]

    def __repr__(self):
        return f"SHRender(sh_degree={self.sh_degree - 1})"
