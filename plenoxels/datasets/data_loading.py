from typing import Tuple, Optional, Dict, Any, List
import logging as log
import os

from tqdm import tqdm
import torch
from torch.multiprocessing import Pool
import torchvision
from PIL import Image


pil2tensor = torchvision.transforms.ToTensor()


def _load_nerf_image_pose(data_dir: str,
                         frame: Dict[str, Any],
                         out_h: Optional[int],
                         out_w: Optional[int],
                         downsample: float,
                         resolution: Tuple[Optional[int], Optional[int]]
                         ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Fix file-path
    f_path = os.path.join(data_dir, frame['file_path'])
    if '.' not in os.path.basename(f_path):
        f_path += '.png'  # so silly...
    if not os.path.exists(f_path):  # there are non-exist paths in fox...
        return (None, None)
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


def _parallel_loader(args):
    torch.set_num_threads(1)
    return _load_nerf_image_pose(**args)


def parallel_load_images(frames: List[Dict[str, Any]],
                         data_dir: str,
                         out_h: Optional[int],
                         out_w: Optional[int],
                         downsample: float,
                         resolution: Tuple[Optional[int], Optional[int]],
                         tqdm_title: str,
                         ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    p = Pool(min(4, len(frames)))
    iterator = p.imap(_parallel_loader,
                      [dict(frame=frame, data_dir=data_dir, downsample=downsample,
                            out_w=out_w, out_h=out_h, resolution=resolution) for frame in frames])
    poses, images = [], []
    for _ in tqdm(range(len(frames)), desc=tqdm_title):
        image, pose = next(iterator)
        if pose is not None:
            poses.append(pose)
        if image is not None:
            images.append(image)
    return images, poses
