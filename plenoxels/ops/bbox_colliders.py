import torch


def intersect_with_aabb(
    near_plane: float, rays_o: torch.Tensor, rays_d: torch.Tensor, aabb: torch.Tensor, training: bool,
):
    """Returns collection of valid rays within a specified near/far bounding box along with a mask
    specifying which rays are valid
    Args:
        rays_o: (num_rays, 3) ray origins
        rays_d: (num_rays, 3) ray directions
        aabb: (2, 3) This is [min point (x,y,z), max point (x,y,z)]
    """
    # avoid divide by zero
    dir_fraction = 1.0 / (rays_d + 1e-6)

    # x
    t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
    # y
    t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
    # z
    t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
    t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

    nears = torch.max(
        torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1
    ).values
    fars = torch.min(
        torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1
    ).values

    # clamp to near plane
    near_plane = near_plane if training else 0
    nears = torch.clamp(nears, min=near_plane)
    fars = torch.maximum(fars, nears + 1e-6)

    return nears, fars
