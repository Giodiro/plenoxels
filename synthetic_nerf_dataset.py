import json
import os

import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import TensorDataset
from tqdm import tqdm


def get_rays(H: int, W: int, focal, c2w) -> torch.Tensor:
    """

    :param H:
    :param W:
    :param focal:
    :param c2w:
    :return:
        Tensor of size [2, W, H, 3] where the first dimension indexes origin and direction
        of rays
    """
    i, j = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing='xy')
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], dim=-1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return torch.stack((rays_o, rays_d), dim=0)


class SyntheticNerfDataset(TensorDataset):
    def __init__(self, datadir, split='train', downsample=1.0, resolution=512, max_frames=None):
        self.datadir = datadir
        self.split = split
        self.img_w: int = int(800 // downsample)
        self.img_h: int = int(800 // downsample)
        self.resolution = resolution
        self.max_frames = max_frames

        self.white_bg = True
        self.near_far = [2.0, 6.0]

        self.scene_bbox = torch.tensor([[-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.pil2tensor = torchvision.transforms.ToTensor()

        self.imgs, self.poses, self.rays = None, None, None
        self.read_meta()

        super().__init__(self.rays, self.imgs)

    def load_image(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path)
        # Low-pass filter
        if self.resolution * 2 < self.img_w:
            img = img.resize((self.resolution * 2, self.resolution * 2), Image.LANCZOS)
        img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
        img = self.pil2tensor(img)       # [4, h, w] (RGBA image)
        img = img.permute(1, 2, 0)       # [h, w, 4]
        img = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])  # Blend A into RGB
        return img

    def read_meta(self):
        with open(os.path.join(self.datadir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
            poses, imgs = [], []
            num_frames = min(len(meta['frames']), self.max_frames or len(meta['frames']))
            for i in tqdm(range(num_frames), desc=f'Loading {self.split} data'):
                frame = meta['frames'][i]
                # Load pose
                pose = np.array(frame['transform_matrix']) @ self.blender2opencv
                poses.append(torch.tensor(pose))
                # Load image
                img_path = os.path.join(
                    self.datadir, self.split, f"{os.path.basename(frame['file_path'])}.png")
                imgs.append(self.load_image(img_path))

            self.imgs = torch.cat(imgs, 0) \
                             .reshape(num_frames * self.img_w * self.img_h, 3)  # [N*H*W, 3]
            self.poses = torch.cat(poses, 0)  # [N, ????]

            # Rays
            focal = 0.5 * self.img_w / np.tan(0.5 * self.meta['camera_angle_x'])
            rays = torch.stack(
                [get_rays(self.img_h, self.img_w, focal, p)
                 for p in self.poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
            # Merge N, H, W dimensions
            rays = rays.permute(1, 0, 2, 3, 4).reshape(2, -1, 3)  # [ro+rd, N*H*W, 3]
            self.rays = rays.to(dtype=torch.float32).contiguous()
