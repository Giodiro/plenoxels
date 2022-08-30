# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from dataclasses import fields, dataclass, make_dataclass
from typing import Optional, List, Tuple, Set, Dict, Iterator
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

__RB_VARIANTS__ = dict()


@dataclass
class RenderBuffer:
    """
    A torch based, multi-channel, pixel buffer object.
    RenderBuffers are "smart" data buffers, used for accumulating tracing results, blending buffers of information,
    and providing discretized images.

    The spatial dimensions of RenderBuffer channels are flexible, e.g: they can be multi-dimensional or flat
    (thus allowing gradual accumulation of traced pixels).

    All RenderBuffer objects support the default rgb, alpha and depth channels.
    Additional custom channels can be specified as **kwargs during construction.
    Access to these channels is identical to the default fields of the RenderBuffer class.
    Customizing how existing and new channels are blended and normalized can be defined via :class:`Channel`
    """

    rgb: Optional[torch.Tensor] = None
    """ rgb is a shaded RGB color. """

    alpha: Optional[torch.Tensor] = None
    """ alpha is the alpha component of RGB-A. """

    depth: Optional[torch.Tensor] = None
    """ depth is usually a distance to the surface hit point."""

    # Renderbuffer supports additional custom channels passed to the Renderbuffer constructor.
    # Some example of custom channels used throughout wisp:
    #     xyz=None,         # xyz is usually the xyz position for the surface hit point.
    #     hit=None,         # hit is usually a segmentation mask of hit points.
    #     normal=None,      # normal is usually the surface normal at the hit point.
    #     shadow =None,     # shadow is usually some additional buffer for shadowing.
    #     ao=None,          # ao is usually some addition buffer for ambient occlusion.
    #     ray_o=None,       # ray_o is usually the ray origin.
    #     ray_d=None,       # ray_d is usually the ray direction.
    #     err=None,         # err is usually some error metric against the ground truth.
    #     gts=None,         # gts is usually the ground truth image.

    def __new__(cls, *args, **kwargs):
        """ If custom channels were specified, create a specialized Renderbuffer class containing them as dataclass
        fields. The returned object is a variant of Renderbuffer whose existance should be transparent to users.

        Args:
            *args: Variable arg of torch.Tensor values for default channels.
            **kwargs: Optional dict of default + optional custom channels, as pairs of [str, Optional[torch.Tensor]].
        """
        # kwargs contains all channels given to the constructor,
        # filter to keep only new channels which aren't listed as default fields under the Renderbuffer class
        class_fields = set(f.name for f in fields(RenderBuffer))
        new_fields = [k for k in kwargs.keys() if k not in class_fields]
        if len(new_fields) > 0:
            # If there are new custom channels, see if a specialized class for this combination of channels
            # have been created before. If so, reuse it.
            class_key = frozenset(new_fields)
            rb_class = __RB_VARIANTS__.get(class_key)
            if rb_class is None:
                # First time this combination of channels is encountered:
                # create a Renderbuffer dataclass variant with new additional channels as fields
                rb_class = make_dataclass(
                    f'Renderbuffer_{len(__RB_VARIANTS__)}',
                    fields=[(k, Optional[torch.Tensor], None) for k in kwargs.keys()],
                    bases=(RenderBuffer,))
                __RB_VARIANTS__[class_key] = rb_class  # Cache for future __new__ calls
            return super(RenderBuffer, rb_class).__new__(rb_class)  # Construct the new Renderbuffer variant
        else:
            # No new fields, just build the default Renderbuffer
            return super(RenderBuffer, cls).__new__(cls)

    def __iter__(self) -> Iterator[Tuple[str, Optional[torch.Tensor]]]:
        """ Creates an iterator on the Renderbuffer fields as {name: tensor}. """
        # A tensor safe version:
        # the dataclasse asdict function performs a deepcopy which does not respect tensors with gradients.
        return iter((f.name, self.get_channel(f.name)) for f in fields(self))

    def __getattr__(self, item):
        """ Invoked when an attribute is not found: used to return a default None for unknown channels. """
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError
        else:
            return None  # Renderbuffer silently returns None when unknown channels are accessed

    @property
    def rgba(self) -> Optional[torch.Tensor]:
        """
        Returns:
            (Optional[torch.Tensor]) A concatenated rgba. If rgb or alpha are none, this property will return None.
        """
        if self.alpha is None or self.rgb is None:
            return None
        else:
            return torch.cat((self.rgb, self.alpha), dim=-1)

    @rgba.setter
    def rgba(self, val: Optional[torch.Tensor]) -> None:
        """
        Args:
            val (Optional[torch.Tensor]) A concatenated rgba channel value, which sets values for the rgb and alpha
            internal channels simultaneously.
        """
        self.rgb = val[..., 0:-1]
        self.alpha = val[..., -1:]

    @property
    def channels(self) -> Set[str]:
        """ Returns a set of channels supported by this RenderBuffer """
        return set([f.name for f in fields(self)])

    def has_channel(self, name: str) -> bool:
        """ Returns whether the RenderBuffer supports the specified channel """
        return name in self.channels

    def get_channel(self, name: str) -> Optional[torch.Tensor]:
        """ Returns the pixels value of the specified channel,
        assuming this RenderBuffer supports the specified channel.
        """
        return getattr(self, name, None)

    @staticmethod
    def _join_fields(rb1: RenderBuffer, rb2: RenderBuffer):
        """ Creates a joint mapping of renderbuffer fields in a format of
            {
                channel1_name: (rb1.c1, rb2.c1),
                channel2_name: (rb1.c2, rb2.c2),
                channel3_name: (rb1.c1, None),  # rb2 doesn't define channel3
            }
            If a renderbuffer does not have define a specific channel, None is returned.
        """
        joint_fields = rb1.channels.union(rb2.channels)
        return {f: (rb1.get_channel(f), rb2.get_channel(f)) for f in joint_fields}

    def _apply(self, fn) -> RenderBuffer:
        """ Applies the function fn on each of the Renderbuffer channels, if not None.
            Returns a new instance with the processed channels.
        """
        data = {}
        for f in fields(self):
            attr = self.get_channel(f.name)
            data[f.name] = None if attr is None else fn(attr)
        return RenderBuffer(**data)

    @staticmethod
    def _apply_on_pair(rb1, rb2, fn) -> RenderBuffer:
        """ Applies the function fn on each of the Renderbuffer channels, if not None.
            Returns a new instance with the processed channels.
        """
        joint_fields = RenderBuffer._join_fields(
            rb1, rb2)  # Union of field names and tuples of values
        combined_channels = map(fn, joint_fields.values())  # Invoke on pair per Renderbuffer field
        return RenderBuffer(
            **dict(zip(joint_fields.keys(), combined_channels)))  # Pack combined fields to a new rb

    def __add__(self, other: RenderBuffer) -> RenderBuffer:
        """ Renderbuffer support spatial acucmulation of pixels (that is, the RenderBuffer is not constrained to
        a 2D grid).
        By default, __add__ performs concatenation of values per channel over the first spatial dimension.
        """
        return self.cat(other)

    def cat(self, other: RenderBuffer, dim: int = 0) -> RenderBuffer:
        """ Concatenates the channels of self and other RenderBuffers.
        If a channel only exists in one of the RBs, that channel will be returned as is.
        For channels that exists in both RBs, the spatial dimensions are assumed to be identical except for the
        concatenated dimension.

        Args:
            other (RenderBuffer) A second buffer to concatenate to the current buffer.
            dim (int): The index of spatial dimension used to concat the channels

        Returns:
            A new RenderBuffer with the concatenated channels.
        """

        def _cat(pair):
            if None not in pair:
                # Concatenating tensors of different dims where one is unsqueezed with dimensionality 1
                if pair[0].ndim == (pair[1].ndim + 1) and pair[0].shape[-1] == 1:
                    pair = (pair[0], pair[1].unsqueeze(-1))
                elif pair[1].ndim == (pair[0].ndim + 1) and pair[1].shape[-1] == 1:
                    pair = (pair[0].unsqueeze(-1), pair[1])
                return torch.cat(pair, dim=dim)
            elif pair[0] is not None and pair[1] is None:  # Channel is None for other but not self
                return pair[0]
            elif pair[0] is None and pair[1] is not None:  # Channel is None for self but not other
                return pair[1]
            else:
                return None

        return RenderBuffer._apply_on_pair(self, other, _cat)

    def transpose(self) -> RenderBuffer:
        """ Permutes dimensions 0 and 1 of each channel.
            The rest of the channel dimensions will remain in the same order.
        """
        return self._apply(lambda x: x.permute(1, 0, *tuple(range(2, x.ndim))))

    def scale(self, size: Tuple, interpolation='bilinear') -> RenderBuffer:
        """ Upsamples or downsamples the renderbuffer pixels using the specified interpolation.
        Scaling assumes renderbuffers with 2 spatial dimensions, e.g. (H, W, C) or (W, H, C).

        Warning: for non-floating point channels, this function will upcast to floating point dtype
        to perform interpolation, and will then re-cast back to the original dtype.
        Hence truncations due to rounding may occur.

        Args:
            size (Tuple): The new spatial dimensions of the renderbuffer.
            interpolation (str): Interpolation method applied to cope with missing or decimated pixels due to
            up / downsampling. The interpolation methods supported are aligned with
            :func:`torch.nn.functional.interpolate`.

        Returns:
            (RenderBuffer): A new RenderBuffer object with rescaled channels.
        """

        def _scale(x):
            assert x.ndim == 3, 'RenderBuffer scale() assumes channels have 2D spatial dimensions.'
            # Some versions of torch don't support direct interpolation of non-fp tensors
            dtype = x.dtype
            if not torch.is_floating_point(x):
                x = x.float()
            x = x.permute(2, 0, 1)[None]
            x = F.interpolate(x, size=size, mode=interpolation)
            x = x[0].permute(1, 2, 0)
            if x.dtype != dtype:
                x = torch.round(x).to(dtype)
            return x

        return self._apply(_scale)

    def numpy_dict(self) -> Dict[str, np.array]:
        """This function returns a dictionary of numpy arrays containing the data of each channel.

        Returns:
            (Dict[str, numpy.Array])
                a dictionary with entries of (channel_name, channel_data)
        """
        _dict = dict(iter(self))
        _dict = {k: v.numpy() for k, v in _dict.items() if v is not None}
        return _dict

    def exr_dict(self) -> Dict[str, torch.Tensor]:
        """This function returns an EXR format compatible dictionary.

        Returns:
            (Dict[str, torch.Tensor])
                a dictionary suitable for use with `pyexr` to output multi-channel EXR images which can be
                viewed interactively with software like `tev`.
                This is suitable for debugging geometric quantities like ray origins and ray directions.
        """
        _dict = self.numpy_dict()
        if 'rgb' in _dict:
            _dict['default'] = _dict['rgb']
            del _dict['rgb']
        return _dict

    def image(self) -> RenderBuffer:
        """This function will return a copy of the RenderBuffer which will contain 8-bit [0,255] images.

        This function is used to output a RenderBuffer suitable for saving as a 8-bit RGB image (e.g. with
        Pillow). Since this quantization operation permanently loses information, this is not an inplace
        operation and will return a copy of the RenderBuffer. Currently this function will only return
        the hit segmentation mask, normalized depth, RGB, and the surface normals.

        If users want custom behaviour, users can implement their own conversion function which takes a
        RenderBuffer as input.
        """
        norm = lambda arr: ((arr + 1.0) / 2.0) if arr is not None else None
        bwrgb = lambda arr: torch.cat([arr] * 3, dim=-1) if arr is not None else None
        rgb8 = lambda arr: (arr * 255.0) if arr is not None else None

        channels = dict()
        if self.rgb is not None:
            channels['rgb'] = rgb8(self.rgb)
        if self.alpha is not None:
            channels['alpha'] = rgb8(self.alpha)
        if self.depth is not None:
            # If the relative depth is respect to some camera clipping plane, the depth should
            # be clipped in advance.
            relative_depth = self.depth / (torch.max(self.depth) + 1e-8)
            channels['depth'] = rgb8(bwrgb(relative_depth))

        # TODO (operel): Write rgba channel
        # TODO (operel): Handle channels in a more general way

        if hasattr(self, 'hit') and self.hit is not None:
            channels['hit'] = rgb8(bwrgb(self.hit))
        else:
            channels['hit'] = None
        if hasattr(self, 'normal') and self.normal is not None:
            channels['normal'] = rgb8(norm(self.normal))
        else:
            channels['normal'] = None

        return RenderBuffer(**channels)

    def reshape(self, *dims: List[int]) -> RenderBuffer:
        """ Reshapes the channels of the renderbuffer to the given dims """
        return self._apply(lambda x: x.reshape(*dims))

    def to(self, *args, **kwargs) -> RenderBuffer:
        """ Shifts the renderbuffer to a new dtype / device """
        return self._apply(lambda x: x.to(*args, **kwargs))

    def cuda(self) -> RenderBuffer:
        """ Shifts the renderbuffer to the default torch cuda device """
        return self._apply(lambda x: x.cuda())

    def cpu(self) -> RenderBuffer:
        """ Shifts the renderbuffer to the torch cpu device """
        return self._apply(lambda x: x.cpu())

    def detach(self) -> RenderBuffer:
        """ Detaches the gradients of all channel tensors of the renderbuffer """
        return self._apply(lambda x: x.detach())

    def byte(self) -> RenderBuffer:
        """ Returns a new RenderBuffer using the byte dtype for each channel """
        return self._apply(lambda x: x.byte())

    def half(self) -> RenderBuffer:
        """ Returns a new RenderBuffer using the half dtype for each channel """
        return self._apply(lambda x: x.half())

    def float(self) -> RenderBuffer:
        """ Returns a new RenderBuffer using the float dtype for each channel """
        return self._apply(lambda x: x.float())

    def double(self) -> RenderBuffer:
        """ Returns a new RenderBuffer using the double dtype for each channel """
        return self._apply(lambda x: x.double())
