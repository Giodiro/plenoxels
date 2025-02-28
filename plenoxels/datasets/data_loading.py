from typing import Tuple, Optional, Dict, Any, List, Sequence
import logging as log
import os
import resource

import torch
from torch.multiprocessing import Pool
import torchvision.transforms
import torchvision.transforms.functional as tf
from PIL import Image
import imageio.v3 as iio

from ..my_tqdm import tqdm

pil2tensor = torchvision.transforms.ToTensor()
# increase ulimit -n (number of open files) otherwise parallel loading might fail
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16192, rlimit[1]))


def _load_llff_image(idx: int,
                     paths: str,
                     data_dir: str,
                     out_h: int,
                     out_w: int,
                     resolution: Tuple[Optional[int], Optional[int]],
                     ) -> torch.Tensor:
    f_path = os.path.join(data_dir, paths[idx])
    img = Image.open(f_path).convert('RGB')

    if resolution[0] is not None and resolution[1] is not None and \
            (resolution[0] * 2 < out_h or resolution[1] * 2 < out_w):
        img = img.resize((resolution[0] * 2, resolution[1] * 2), Image.LANCZOS)
        img = img.resize((out_w, out_h), Image.LANCZOS)
    else:
        img = img.resize((out_w, out_h), Image.LANCZOS)
    img = pil2tensor(img)  # [C, H, W]
    img = img.permute(1, 2, 0)  # [H, W, C]
    return img


def _parallel_loader_llff_image(args):
    torch.set_num_threads(1)
    return _load_llff_image(**args)


def _load_nerf_image_pose(idx: int,
                          frames: List[Dict[str, Any]],
                          data_dir: str,
                          out_h: Optional[int],
                          out_w: Optional[int],
                          downsample: float,
                          resolution: Tuple[Optional[int], Optional[int]]
                          ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    # Fix file-path
    f_path = os.path.join(data_dir, frames[idx]['file_path'])
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
        log.warning("360 non-square")
    if resolution[0] is not None and resolution[1] is not None and \
            (resolution[0] * 2 < out_h or resolution[1] * 2 < out_w):
        img = img.resize((resolution[0] * 2, resolution[1] * 2), Image.LANCZOS)
        img = img.resize((out_w, out_h), Image.LANCZOS)
    else:
        img = img.resize((out_w, out_h), Image.LANCZOS)
    img = pil2tensor(img)  # [C, H, W]
    img = img.permute(1, 2, 0)  # [H, W, C]

    pose = torch.tensor(frames[idx]['transform_matrix'], dtype=torch.float32)

    return (img, pose)


def _parallel_loader_nerf_image_pose(args):
    torch.set_num_threads(1)
    return _load_nerf_image_pose(**args)


def _load_video_1cam(idx: int,
                     paths: List[str],
                     poses: torch.Tensor,
                     out_h: int,
                     out_w: int,
                     load_every: int = 1
                     ):  # -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    filters = [
        ("scale", f"w={out_w}:h={out_h}")
    ]
    all_frames = iio.imread(
        paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,
        filter_sequence=filters,)
    imgs, timestamps = [], []
    for frame_idx, frame in enumerate(all_frames):
        if frame_idx % load_every != 0:
            continue
        if frame_idx >= 300:  # Only look at the first 10 seconds
            break
        # Frame is np.ndarray in uint8 dtype (H, W, C)
        imgs.append(
            torch.from_numpy(frame)  #.to(torch.float32).div(255),
        )
        timestamps.append(frame_idx)
    imgs = torch.stack(imgs, 0)
    med_img, _ = torch.median(imgs, dim=0)  # [h, w, 3]
    return (imgs,
            poses[idx].expand(len(timestamps), -1, -1),
            med_img,
            torch.tensor(timestamps, dtype=torch.int32))


def _parallel_loader_video(args):
    torch.set_num_threads(1)
    return _load_video_1cam(**args)


def parallel_load_images(tqdm_title,
                         dset_type: str,
                         num_images: int,
                         **kwargs) -> List[Any]:
    max_threads = 4
    if dset_type == 'llff':
        fn = _parallel_loader_llff_image
    elif dset_type == 'synthetic':
        fn = _parallel_loader_nerf_image_pose
    elif dset_type == 'video':
        fn = _parallel_loader_video
        # giac: Can increase to e.g. 10 if loading 4x subsampled images. Otherwise OOM.
        max_threads = 4
    else:
        raise ValueError(dset_type)
    p = Pool(min(max_threads, num_images))

    iterator = p.imap(fn, [{"idx": i, **kwargs} for i in range(num_images)])
    outputs = []
    for _ in tqdm(range(num_images), desc=tqdm_title):
        out = next(iterator)
        if out is not None:
            outputs.append(out)
    return outputs
