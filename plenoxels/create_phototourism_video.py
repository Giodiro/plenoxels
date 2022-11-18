import sys
import math
from collections import defaultdict
import logging as log

import numpy as np
from plenoxels.models.lowrank_appearance import LowrankAppearance
import torch
from tqdm import trange
from pathlib import Path
import glob
import os
import pandas as pd

from plenoxels.models.lowrank_video import LowrankVideo
from plenoxels.datasets.video_datasets import load_llffvideo_poses
from plenoxels.datasets.ray_utils import gen_camera_dirs, normalize, average_poses, viewmatrix
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.datasets.photo_tourism import get_rays_tourism

train_images = {"sacre" : 1179,
            "trevi" : 1689,
            "brandenburg" : 763}

test_images = {"sacre" : 21,
            "trevi" : 19,
            "brandenburg" : 10}

dataset = "brandenburg"

log.basicConfig(level=log.INFO,
                format='%(asctime)s|%(levelname)8s| %(message)s',
                handlers=[log.StreamHandler(sys.stdout)],
                force=True)


def generate_spiral_path(poses: np.ndarray,
                         near_fars: np.ndarray,
                         n_frames=120,
                         n_rots=0.8,
                         zrate=.5) -> np.ndarray:
    # center pose
    if poses.shape[1] > 3:
        poses = poses[:,0:3,:]
    c2w = average_poses(poses)  # [3, 4]

    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = np.min(near_fars) * 0.1, np.max(near_fars) * 5.0
    dt = 0.75
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = np.concatenate([radii, [1.]])

    # Generate poses for spiral path.
    render_poses = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
        # t = radii * [np.cos(theta), np.sin(theta), -np.sin(theta * zrate), 1.]
        rotation = c2w[:3,:3]
        translation = c2w[:,3:4] + np.array([[0.1*np.cos(theta), -0.05-0.01*np.sin(theta), -0.2+0.2*np.sin(theta * zrate)]]).T
        pose = np.concatenate([rotation, translation],axis=1)
        render_poses.append(pose)
    return np.stack(render_poses, axis=0)


def eval_step(data, model, batch_size=8192):
    """
    Note that here `data` contains a whole image. we need to split it up before tracing
    for memory constraints.
    """
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            rays_o = data["rays_o"].squeeze()
            rays_d = data["rays_d"]
            near_far = data["near_far"]
            timestamp = data["timestamps"]
            
            preds = defaultdict(list)
            num_batches = math.ceil(rays_o.shape[0] / batch_size)
            for b in trange(num_batches):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].contiguous().cuda()
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].contiguous().cuda()
                timestamps_b = timestamp[b * batch_size: (b + 1) * batch_size].contiguous().cuda()
                near_far_b = near_far[b * batch_size: (b + 1) * batch_size].contiguous().cuda()

                outputs = model(
                    rays_o_b, rays_d_b, timestamps_b, near_far=near_far_b,
                    channels={"rgb", "depth"},
                    bg_color=torch.tensor([1.0]*3).cuda(),
                )
                for k, v in outputs.items():
                    if not isinstance(v, list):
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}


def load_data(datadir, num_frames, H, W):
    tsv = glob.glob(os.path.join(datadir, '*.tsv'))[0]
    files = pd.read_csv(tsv, sep='\t')
    files = files[~files['id'].isnull()] # remove data without id
    files.reset_index(inplace=True, drop=True)
    files = files[files["split"]=='train']
    imagepaths = sorted((Path(datadir) / "dense" / "images").glob("*.jpg"))
    imkey = np.array([os.path.basename(im) for im in imagepaths])
    idx = np.in1d(imkey, files["filename"])

    poses = np.load(Path(datadir) / "c2w_mats.npy")[idx]
    kinvs = np.load(Path(datadir) / "kinv_mats.npy")[idx]
    bounds = np.load(Path(datadir) / "bds.npy")[idx]
    kinv = torch.from_numpy(kinvs[0]).to(torch.float32) # Just pick one camera intrinsic

    scale = 0.05
    poses[:, :3, 3:4] = poses[:, :3, 3:4] * scale 
    poses = torch.tensor(poses)[:, :3, :].float()
    bounds = bounds * np.array([0.9, 1.2]) * scale

    spiral_poses = generate_spiral_path(
        poses=poses,
        near_fars=bounds,
        n_frames=num_frames
    )
    spiral_poses = torch.from_numpy(spiral_poses).float()
    bounds = torch.from_numpy(bounds).float()

    rays_o = []
    rays_d = []
    timestamps = []
    near_fars = []
    # interp_time = torch.tensor(0)
    # torch.linspace(0, 1708, dtype=torch.int)
    for pose_id in range(spiral_poses.shape[0]):

        c2w = spiral_poses[pose_id]  # [3, 4]

        origins, directions = get_rays_tourism(H, W, kinv, c2w)

        origins = origins.reshape(-1, 3)
        directions = directions.reshape(-1, 3)

        rays_o.append(origins)
        rays_d.append(directions)
        # timestamps.append(interp_time.repeat(origins.shape[0]))
        timestamps.append(torch.tensor(100 + pose_id / spiral_poses.shape[0]).repeat(origins.shape[0]))
        # Find the closest cam TODO: This is the crappiest way to calculate distance between cameras!
        # TODO: continue updating for phototourism here
        closest_cam_idx = torch.linalg.norm(
            poses.view(poses.shape[0], -1) - c2w.view(-1), dim=1).argmin()
        
        near_fars.append((bounds[closest_cam_idx] + torch.tensor([0.05, 0.0])).repeat(origins.shape[0], 1))
    
    rays_o = torch.cat(rays_o, 0)
    rays_d = torch.cat(rays_d, 0)
    timestamps = torch.cat(timestamps, 0)
    near_fars = torch.cat(near_fars, 0)


    data = {
        "rays_o": rays_o,
        "rays_d": rays_d,
        "near_far": near_fars,
        "timestamps": timestamps,
    }
    log.info(f"Loaded {rays_o.shape[0]} rays")
    return data


def load_model(checkpoint_path):
    m_data = torch.load(checkpoint_path)

    reso = [
        m_data['model']['grids.0.0'].shape[-1],
        m_data['model']['grids.0.0'].shape[-2],
        m_data['model']['grids.0.1'].shape[-2],
    ]
    log.info("Will load model with resolution: %s" % (reso, ))

    model = LowrankAppearance(
        aabb=torch.tensor([[-2., -2., -2.], [2., 2., 2.]]),
        len_time=train_images[dataset] + test_images[dataset],
        is_ndc=False,
        is_contracted=True,
        lookup_time=False,
        proposal_sampling=True,
        global_scale=m_data['model']['spatial_distortion.global_scale'],
        global_translation=m_data['model']['spatial_distortion.global_translation'],
        raymarch_type='fixed',
        single_jitter=False,
        n_intersections=48,
        density_activation='trunc_exp',
        use_F=False,
        sh=False,
        density_field_resolution=[128, 256],
        density_field_rank=1,
        num_proposal_samples=[256, 96],
        proposal_feature_dim=10,
        proposal_decoder_type='nn',
        density_model='triplane',
        multiscale_res=[1, 2, 4, 8],
        grid_config=[
            {
                "input_coordinate_dim": 3,
                "output_coordinate_dim": 32,
                "grid_dimensions": 2,
                "resolution": reso,
                "rank": 2,
                "time_reso" : test_images[dataset] + train_images[dataset],
            }
        ],
    )

    model.load_state_dict(m_data['model'])
    model.cuda()
    log.info("Loaded model")
    return model


def save_video(out_file, spiral_outputs, output_key='rgb'):
    imgs = spiral_outputs[output_key]

    image_len = 800 * 800
    num_images = imgs.shape[0] // image_len
    log.info("Output contains %d frames" % (num_images, ))

    frames = (
        (imgs.view(num_images, 800, 800, 3) * 255.0)
        .to(torch.uint8)
    ).cpu().detach().numpy()
    frames = [frames[i] for i in range(num_images)]
    write_video_to_file(out_file, frames)


def run():
    datadir = '/home/warburg/data/phototourism/brandenburg'
    checkpoint_path = '/home/sfk/plenoxels/logs/phototourism/brandenburg_cvpr/model.pth'
    output_path = '/home/sfk/plenoxels/logs/phototourism/brandenburg_cvpr/test_video.mp4'
    num_frames = 200

    data = load_data(datadir, num_frames, H=800, W=800)
    model = load_model(checkpoint_path)
    spiral_outputs = eval_step(data, model, batch_size=8192 * 2)
    save_video(output_path, spiral_outputs, output_key='rgb')


if __name__ == "__main__":
    run()
