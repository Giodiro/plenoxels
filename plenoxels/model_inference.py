import os
import sys
import torch
import math
import itertools
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join('..')))
import cv2
import numpy as np
from plenoxels.models.lowrank_appearance import LowrankAppearance
from plenoxels.datasets.photo_tourism import PhotoTourismDataset
from plenoxels.ops.image import metrics
from plenoxels.models.utils import grid_sample_wrapper, compute_plane_tv, compute_line_tv, raw2alpha

#checkpoint_path = "logs/5nov_trevi/hexplane_lr001_tv0_histloss01_proposal128x256_256x96_ninsect48l1_appearance_planes_reg0.1/model.pth"
#save_dir = "logs/trevi_rank_vis/hexplane_lr001_tv0_histloss01_proposal128x256_256x96_ninsect48l1_appearance_planes_reg0.1/"

#checkpoint_path = "logs/trevi/nov8/with_nn_tv1e3_l1time001_rank1_outdim32/model.pth"
#save_dir = "logs/trevi/nov8/with_nn_tv1e3_l1time001_rank1_outdim32/"

checkpoint_path =  "logs/brandenburg/nov10/hexplane_lr005_tv0_rank2_l101_outdim32_scales1_2_4_8_appearance_code32_colornet3/model.pth"
save_dir = "logs/brandenburg/nov10/hexplane_lr005_tv0_rank2_l101_outdim32_scales1_2_4_8_appearance_code32_colornet3/"

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
                if not isinstance(v, list):
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

#grid_config = '[{"input_coordinate_dim": 4, "output_coordinate_dim": 32, "grid_dimensions" : 2, "resolution": [80, 40, 20, 1708], "rank": 1, "time_reso" : 1708}, {"input_coordinate_dim": 5, "resolution": [6, 6, 6, 6, 6], "feature_dim": 32, "init_std": 0.001}]'
grid_config = '[{"input_coordinate_dim": 4, "output_coordinate_dim": 32, "grid_dimensions" : 2, "resolution": [80, 40, 20, 773], "rank": 2, "time_reso" : 773}, {"input_coordinate_dim": 5, "resolution": [6, 6, 6, 6, 6], "feature_dim": 32, "init_std": 0.001}]'
model = LowrankAppearance(grid_config, 
                aabb=torch.tensor([[-2., -2., -2.], [2., 2., 2.]]), 
                len_time=763+710, 
                is_ndc=False, 
                is_contracted=True, 
                lookup_time=True, 
                sh=False, 
                use_F=False, 
                n_intersections=48 ,
                num_samples_multiplier=2,
                raymarch_type = "fixed",
                spacing_fn = "linear",
                single_jitter= False,
                density_activation="trunc_exp",
                proposal_sampling=True,
                num_proposal_samples=[256, 96],
                density_field_rank = 1,
                density_field_resolution = [128, 256],
                density_model = 'hexplane',
                multiscale_res =[ 1, 2, 4, 8],
                proposal_feature_dim=10,
                proposal_decoder_type= "nn",
                color_net=3,
                appearance_code_size=32,
                )

model = model.cuda()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])

# store original parameters
parameters = []
for multires_grids in model.grids:
    parameters.append([grid.data for grid in multires_grids])

"""
for plane_idx, grid in enumerate(model.grids):
                    
    _, c, h, w = grid.data.shape
    rank = model.config[0]["rank"][0]
    dim = model.config[0]["output_coordinate_dim"]
    grid = grid.data.view(dim, rank, h, w)
    for r in range(rank):
        
        density = grid[-1, r, :, :]
        density = ((density - density.min())/(density.max() - density.min()) * 255).cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"density-plane-{plane_idx}-rank_{r}.png"), density)
    
        rays_d = torch.ones((h*w, 3), device=grid.device)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        features = grid[:, r, :, :].view(dim, h*w).permute(1,0)
        color = model.decoder.compute_color(features, rays_d) * 255
        color = color.view(h, w, 3).cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"color-plane-{plane_idx}-rank_{r}.png"), color[:,:,::-1])
"""

# visualize time grid
train_dataset = PhotoTourismDataset("/work3/frwa/data/phototourism/brandenburg", "train", debug=True)
train_dataset.training = False
rank_idx = "avg"
with torch.no_grad():
    for img_idx, data in enumerate(train_dataset):
        print("==> im idx ", img_idx)
        
        
        for i in range(len(model.grids)):
            # turn spatial grids off
            for plane_idx in [0, 1, 3]:
                model.grids[i][plane_idx].data = torch.ones_like(parameters[i][plane_idx])
            
            # turn time grids on        
            for plane_idx in [2, 4, 5]:
                model.grids[i][plane_idx].data = parameters[i][plane_idx]
            
        data["timestamps"] = data["timestamps"] - train_dataset.n_train_images 
        preds = eval_step(data, model)
        summary, out_img = evaluate_metrics(data["imgs"], preds, dset=train_dataset, img_idx=img_idx)
        cv2.imwrite(f"{save_dir}{img_idx}_time.png", out_img)

        
        for i in range(len(model.grids)):    
            # turn time grids off    
            for plane_idx in [2, 4, 5]:
                model.grids[i][plane_idx].data = torch.ones_like(parameters[i][plane_idx])
                
            # turn spatial grids on        
            for plane_idx in [0, 1, 3]:
                model.grids[i][plane_idx].data = parameters[i][plane_idx]
                
        preds = eval_step(data, model)
        summary, out_img = evaluate_metrics(data["imgs"], preds, dset=train_dataset, img_idx=img_idx)
        cv2.imwrite(f"{save_dir}{img_idx}_spatial.png", out_img[:, :, ::-1])
    
        if img_idx >= 30:
            break
        
#visualize rank

test_dataset = PhotoTourismDataset("/work3/frwa/data/phototourism/trevi", "test")
with torch.no_grad():
    for img_idx, data in enumerate(test_dataset):
        #for rank_idx in ranks:
        preds = eval_step(data, model)
        summary, out_img = evaluate_metrics(data["imgs"], preds, dset=test_dataset, img_idx=img_idx)
        cv2.imwrite(f"{save_dir}{img_idx}_{rank_idx}.png",  out_img[:, :, ::-1])



