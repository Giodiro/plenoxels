import jax
import jax.numpy as jnp


@jax.jit
def grid_lookup(x, y, z, grid):
    indices, data = grid
    ret = [jnp.where(indices[x, y, z, jnp.newaxis] >= 0, d[indices[x, y, z]], jnp.zeros(3)) for d in
           data[:-1]]
    ret.append(jnp.where(indices[x, y, z] >= 0, data[-1][indices[x, y, z]], 0))
    return ret


def vectorize(index, resolution):
    i = index // (resolution ** 2)
    j = (index - i * resolution * resolution) // resolution
    k = index - i * resolution * resolution - j * resolution
    return jnp.array([i, j, k])


def scalarize(i, j, k, resolution):
    return i * resolution * resolution + j * resolution + k


class EMA():
    def __init__(self, weighting=0.9):
        self.weighting = weighting
        self.val = None

    def update(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.weighting * val + (1 - self.weighting) * self.val

    @property
    def value(self):
        return self.val
