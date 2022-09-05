from typing import Tuple, Optional, Dict, Any, List, Sequence
import logging as log
import os

from tqdm import tqdm
import torch
from torch.multiprocessing import Pool
import torchvision
from PIL import Image

pil2tensor = torchvision.transforms.ToTensor()


def _load_llff_image(data_dir: str,
                     path: str,
                     out_h: int,
                     out_w: int,
                     resolution: Tuple[Optional[int], Optional[int]],
                     ) -> torch.Tensor:
    f_path = os.path.join(data_dir, path)
    img = Image.open(f_path).convert('RGB')

    if resolution[0] is not None and resolution[1] is not None and \
            (resolution[0] * 2 < out_h or resolution[1] * 2 < out_w):
        img = img.resize((resolution[0] * 2, resolution[1] * 2), Image.LANCZOS)
        img = img.resize((out_h, out_w), Image.LANCZOS)
    else:
        img = img.resize((out_h, out_w), Image.LANCZOS)
    img = pil2tensor(img)  # [C, H, W]
    img = img.permute(1, 2, 0)  # [H, W, C]
    return img


def _parallel_loader_llff_image(args):
    torch.set_num_threads(1)
    return _load_llff_image(**args)


def _load_nerf_image_pose(data_dir: str,
                          frame: Dict[str, Any],
                          out_h: Optional[int],
                          out_w: Optional[int],
                          downsample: float,
                          resolution: Tuple[Optional[int], Optional[int]]
                          ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    # Fix file-path
    f_path = os.path.join(data_dir, frame['file_path'])
    if '.' not in os.path.basename(f_path):
        f_path += '.png'  # so silly...
    if not os.path.exists(f_path):  # there are non-exist paths in fox...
        return None
    img = Image.open(f_path)
    if out_h is None:
        out_h = int(img.size[0] / downsample)
    if out_w is None:
        out_w = int(img.size[1] / downsample)
    # Now we should downsample to out_h, out_w and low-pass filter to resolution * 2.
    # We only do the low-pass filtering if resolution * 2 is lower-res than out_h, out_w
    if out_h != out_w:
        log.warning("")
    if resolution[0] is not None and resolution[1] is not None and \
            (resolution[0] * 2 < out_h or resolution[1] * 2 < out_w):
        img = img.resize((resolution[0] * 2, resolution[1] * 2), Image.LANCZOS)
        img = img.resize((out_h, out_w), Image.LANCZOS)
    else:
        img = img.resize((out_h, out_w), Image.LANCZOS)
    img = pil2tensor(img)  # [C, H, W]
    img = img.permute(1, 2, 0)  # [H, W, C]

    pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32)

    return (img, pose)


def _parallel_loader_nerf_image_pose(args):
    torch.set_num_threads(1)
    return _load_nerf_image_pose(**args)


def parallel_load_images(image_iter: Sequence[Any],
                         tqdm_title,
                         dset_type: str,
                         **kwargs) -> List[Any]:
    p = Pool(min(4, len(image_iter)))
    if dset_type == 'llff':
        fn = _parallel_loader_llff_image
        iter_name = "path"
    elif dset_type == 'synthetic':
        fn = _parallel_loader_nerf_image_pose
        iter_name = "frame"
    else:
        raise ValueError(dset_type)

    iterator = p.imap(fn, [{iter_name: img_desc, **kwargs} for img_desc in image_iter])
    outputs = []
    for _ in tqdm(range(len(image_iter)), desc=tqdm_title):
        out = next(iterator)
        if out is not None:
            outputs.append(out)
    return outputs
