import collections

from torch.utils.data.dataloader import default_collate

from .transforms import SampleRays
from .multiview_dataset import NerfDataset
from ..core import Rays

__all__ = (
    "NerfDataset",
    "SampleRays",
    "ray_default_collate",
)


def ray_default_collate(batch):
    elem = batch[0]

    if isinstance(elem, Rays):
        return Rays.stack(batch)
    elif isinstance(elem, collections.abc.Mapping):  # noqa
        return {key: ray_default_collate([d[key] for d in batch]) for key in elem}
    else:
        return default_collate(batch)
