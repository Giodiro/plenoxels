import jax
import jax.numpy as jnp


# Based on https://github.com/google-research/google-research/blob/d0a9b1dad5c760a9cfab2a7e5e487be00886803c/jaxnerf/nerf/model_utils.py#L166
def volumetric_rendering(rgb, sigma, z_vals, dirs, white_bkgd=True):
    """Volumetric Rendering Function.
    Args:
      rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
      sigma: jnp.ndarray(float32), density, [batch_size, num_samples].
      z_vals: jnp.ndarray(float32), [batch_size, num_samples].
      dirs: jnp.ndarray(float32), [batch_size, 3].
      white_bkgd: bool.
    Returns:
      comp_rgb: jnp.ndarray(float32), [batch_size, 3].
      disp: jnp.ndarray(float32), [batch_size].
      acc: jnp.ndarray(float32), [batch_size].
      weights: jnp.ndarray(float32), [batch_size, num_samples]
  """
    eps = 1e-10
    dists = z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1]
    print("z_vals", z_vals.shape)
    dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :],
                                    axis=-1)  # Convert ray-relative distance to absolute distance (shouldn't matter if rays_d is normalized)
    print("dists", dists.shape)
    # Note that we're quietly turning sigma from [..., 0] to [...].
    alpha = 1.0 - jnp.exp(
        -jax.nn.relu(sigma) * dists)  # What fraction of light gets stuck in each voxel
    print("alpha", alpha.shape)
    accum_prod = jnp.concatenate([
        jnp.ones_like(alpha[Ellipsis, :1], alpha.dtype),
        jnp.cumprod(1.0 - alpha[Ellipsis, :-1] + eps, axis=-1)
    ],
        axis=-1)  # How much light is left as we enter each voxel
    print("accum_prod", accum_prod.shape)
    weights = alpha * accum_prod  # The absolute amount of light that gets stuck in each voxel
    comp_rgb = (weights[Ellipsis, None] * jax.nn.sigmoid(rgb)).sum(
        axis=-2)  # Accumulated color over the samples, ignoring background
    depth = (weights * z_vals[Ellipsis, :-1]).sum(
        axis=-1)  # Weighted average of depths by contribution to final color
    acc = weights.sum(axis=-1)  # Total amount of light absorbed along the ray
    # Equivalent to (but slightly more efficient and stable than):
    #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
    inv_eps = 1 / eps
    disp = acc / depth
    disp = jnp.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp,
                     inv_eps)  # disparity = inverse depth
    if white_bkgd:
        comp_rgb = comp_rgb + (
                1. - acc[Ellipsis, None])  # Including the white background in the final color
    return comp_rgb, disp, acc, weights
