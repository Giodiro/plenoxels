import os
import sys
import torch
import math
import itertools
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join('..')))
import cv2
from plenoxels.models.lowrank_appearance import LowrankAppearance
from plenoxels.datasets.photo_tourism import PhotoTourismDataset
from plenoxels.ops.image import metrics
from plenoxels.models.utils import grid_sample_wrapper, compute_plane_tv, compute_line_tv, raw2alpha

checkpoint_path = "logs/trevi/tv_0.05_appearance_code_wo_density_longer_test_time_optim/model.pth"
save_dir = "logs/trevi_rank_vis/05_appearance_code_wo_density_longer_test_time_optim/"
os.makedirs(save_dir, exist_ok=True)

ranks = [0, 1, "avg"]
def eval_step(data, model, batch_size=4096):
    """
    Note that here `data` contains a whole image. we need to split it up before tracing
    for memory constraints.
    """
    
    with torch.cuda.amp.autocast(enabled=True):
        rays_o = data["rays_o"]
        rays_d = data["rays_d"]
        near_far = data["near_far"].cuda() if data["near_far"] is not None else None
        timestamp = data["timestamps"]

        if rays_o.ndim == 3:
            rays_o = rays_o.squeeze(0)
            rays_d = rays_d.squeeze(0)

        preds = defaultdict(list)
        for b in range(math.ceil(rays_o.shape[0] / batch_size)):
            rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].cuda()
            rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].cuda()
            timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).cuda()

            outputs = model(rays_o_b, rays_d_b, timestamps_d_b, channels={"rgb", "depth"}, bg_color=1, near_far=near_far)
            
            for k, v in outputs.items():
                preds[k].append(v)
    return {k: torch.cat(v, 0) for k, v in preds.items()}


def evaluate_metrics(gt, preds, dset, img_idx):
    if isinstance(dset.img_h, int):
        img_h, img_w = dset.img_h, dset.img_w
    else:
        img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]

    preds_rgb = preds["rgb"].reshape(img_h, img_w, 3).cpu()
    exrdict = {
        "preds": preds_rgb.numpy(),
    }
    summary = dict()

    if "depth" in preds:
        # normalize depth and add to exrdict
        depth = preds["depth"]
        depth = depth - depth.min()
        depth = depth / depth.max()
        depth = depth.cpu().reshape(img_h, img_w)[..., None]
        preds["depth"] = depth
        exrdict["depth"] = preds["depth"].numpy()

    if gt is not None:
        gt = gt.reshape(img_h, img_w, -1).cpu()
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
        
        # if phototourism then only compute metrics on the right side of the image
        if hasattr(model, "appearance_code"):
            mid = gt.shape[1] // 2
            gt_right = gt[:, mid:]
            preds_rgb_right = preds_rgb[:, mid:]
            
            err = (gt_right - preds_rgb_right) ** 2
            exrdict["err"] = err.numpy()
            summary["mse"] = torch.mean(err)
            summary["psnr"] = metrics.psnr(preds_rgb_right, gt_right)
            summary["ssim"] = metrics.ssim(preds_rgb_right, gt_right)
        else:
            err = (gt - preds_rgb) ** 2
            exrdict["err"] = err.numpy()
            summary["mse"] = torch.mean(err)
            summary["psnr"] = metrics.psnr(preds_rgb, gt)
            summary["ssim"] = metrics.ssim(preds_rgb, gt)

    out_img = preds_rgb
    if "depth" in preds:
        out_img = torch.cat((out_img, preds["depth"].expand_as(out_img)))
    out_img = (out_img * 255.0).byte().numpy()

    return summary, out_img


def compute_features(self,
                         pts,
                         timestamps,
                         return_coords: bool = False
                         ):
    grid_space = self.grids  # space: 3 x [1, rank * F_dim, reso, reso]
    grid_time = self.time_coef  # time: [rank * F_dim, time_reso]
    level_info = self.config[0]  # Assume the first grid is the index grid, and the second is the feature grid

    dim = level_info["output_coordinate_dim"] - 1 if level_info["output_coordinate_dim"] == 28 else level_info["output_coordinate_dim"]

    interp_time = grid_time[:, timestamps.long()].unsqueeze(0).repeat(pts.shape[0], 1)  # [n, F_dim * rank]
    interp_time = interp_time.view(-1, dim, level_info["rank"][0])  # [n, F_dim, rank]
    
    # add density one to appearance code
    if level_info["output_coordinate_dim"] == 28:
            interp_time = torch.cat([interp_time, torch.ones_like(interp_time[:, 0:1, :])], dim=1)
    
    # Interpolate in space
    interp = pts
    coo_combs = list(itertools.combinations(
        range(interp.shape[-1]),
        level_info.get("grid_dimensions", level_info["input_coordinate_dim"])))
    interp_space = None  # [n, F_dim, rank]
    for ci, coo_comb in enumerate(coo_combs):
        if interp_space is None:
            interp_space = (
                grid_sample_wrapper(grid_space[ci], interp[..., coo_comb]).view(
                    -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
        else:
            interp_space = interp_space * (
                grid_sample_wrapper(grid_space[ci], interp[..., coo_comb]).view(
                    -1, level_info["output_coordinate_dim"], level_info["rank"][ci]))
    # Combine space and time over rank
    interp = (interp_space * interp_time)  # [n, F_dim]
    print("==> rank_idx ", rank_idx)
    if rank_idx == "avg":
        out = interp.mean(dim=-1)  # [n, F_dim]
    else:
        out = interp[:, :, rank_idx]
    return out

LowrankAppearance.compute_features = compute_features

grid_config = '[{"input_coordinate_dim": 4, "output_coordinate_dim": 28, "grid_dimensions" : 2, "resolution": [512, 512, 512, 1708], "rank": 2, "time_reso" : 1708}, {"input_coordinate_dim": 5, "resolution": [6, 6, 6, 6, 6], "feature_dim": 28, "init_std": 0.001}]'
model = LowrankAppearance(grid_config, 
                aabb=torch.tensor([[-2., -2., -2.], [2., 2., 2.]]), 
                len_time=1708, 
                is_ndc=False, 
                is_contracted=True, 
                lookup_time=True, 
                sh=True, 
                use_F=False, 
                n_intersections=400 ,
                num_sample_multiplier=2,
                 raymarch_type = "fixed",
                 spacing_fn = "linear",
                 single_jitter= False)

model = model.cuda()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])

test_dataset = PhotoTourismDataset("/work3/frwa/data/phototourism/trevi", "test")
train_dataset = PhotoTourismDataset("/work3/frwa/data/phototourism/trevi", "train")

with torch.no_grad():
    for img_idx, data in enumerate(test_dataset):
        for rank_idx in ranks:
            preds = eval_step(data, model)
            summary, out_img = evaluate_metrics(data["imgs"], preds, dset=test_dataset, img_idx=img_idx)
            cv2.imwrite(f"{save_dir}{img_idx}_{rank_idx}.png", out_img)

import pdb; pdb.set_trace()
