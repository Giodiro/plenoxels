import math
import os
from typing import Tuple, Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from nerfacc import OccupancyGrid, ray_marching, rendering

__all__ = (
    "get_freer_gpu",
    "plot_ts",
    "test_model",
    "user_ask_options",
    "get_cosine_schedule_with_warmup",
    "get_step_schedule_with_warmup",
    "render_image",
    "init_dloader_random",
)


def get_freer_gpu():
    try:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)
    except:  # On some Giacomo GPUs this fails due to newer drivers. But I have 1 GPU anyways
        return 0


def save_image(img_or_fig, log_dir, img_name, iteration, summary_writer):
    if log_dir is not None:
        out_path = os.path.join(log_dir, f"{img_name}_iter{iteration}.png")
        if isinstance(img_or_fig, plt.Figure):
            img_or_fig.savefig(out_path)
            plt.close(img_or_fig)
        else:
            # imageio wants channels last
            imageio.imwrite(out_path, img_or_fig)
    if summary_writer is not None:
        if isinstance(img_or_fig, plt.Figure):
            summary_writer.add_figure(img_name, img_or_fig, global_step=iteration)
            plt.close(img_or_fig)
        else:
            summary_writer.add_image(img_name, img_or_fig, global_step=iteration, dataformats="HWC")


def render_ts_img(data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  model,
                  batch_size: int,
                  img_h: int, img_w: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    rays_o, rays_d, rgb = data
    preds, depths = [], []
    for b in range(math.ceil(rays_o.shape[0] / batch_size)):
        rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to("cuda")
        rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to("cuda")
        output = model(rays_o_b, rays_d_b)
        if isinstance(output, torch.Tensor):
            preds.append(output.cpu())
        elif isinstance(output, tuple):
            preds.append(output[0].cpu())
            if len(output) > 1 and output[1] is not None:
                depths.append(output[1].cpu())
    pred = torch.cat(preds, 0).view(img_h, img_w, 3)
    depth = None
    if len(depths) > 0:
        depth = torch.cat(depths, 0).view(img_h, img_w)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3] * rgb[..., 3:] + (1.0 - rgb[..., 3:])
    rgb = rgb.view(img_h, img_w, 3)
    return pred, rgb, depth


def plot_ts(ts_dset, dset_id, renderer, log_dir, iteration, batch_size=10_000, image_id=0,
            verbose=True, summary_writer=None, render_fn=None, plot_type="imageio"):
    if render_fn is None:
        def render_fn(ro, rd):
            return renderer(ro, rd, dset_id)[0]
    with torch.autograd.no_grad():
        ts_el = ts_dset[image_id]
        pred, rgb, opt_depth = render_ts_img(
            ts_el, model=render_fn, batch_size=batch_size, img_h=ts_dset.img_h, img_w=ts_dset.img_w)
        mse = torch.mean((pred - rgb) ** 2)
        psnr = -10.0 * torch.log(mse) / math.log(10)
    if verbose:
        print(f"D{dset_id} Test PSNR={psnr:.2f}")
    if log_dir is not None:
        if plot_type == "imageio":
            fig = torch.cat((torch.clamp(pred, 0, 1), rgb), dim=1)
            fig = (fig * 255).numpy().astype(np.uint8)
        elif plot_type == "matplotlib":
            fig, ax = plt.subplots(ncols=2 if opt_depth is None else 3, figsize=(15, 8), dpi=90)
            ax[0].imshow(torch.clamp(pred, 0, 1))
            ax[0].axis('off')
            err = ((pred - rgb)**2).sum(-1)
            err = (err - err.min()) / (err.max() - err.min())  # normalize in 0,1
            ax[1].imshow(err, cmap="plasma")
            ax[1].axis('off')
            if opt_depth is not None:
                ax[2].imshow((opt_depth - opt_depth.min()) / (opt_depth.max() - opt_depth.min()), cmap="plasma")
                ax[2].axis('off')
            ax[0].set_title(f"PSNR={psnr:.2f}")
        else:
            raise ValueError(f"plot-type {plot_type} invalid. "
                             f"Accepted values are 'imageio' and 'matplotlib'.")
        save_image(fig, log_dir, f"dset-{dset_id}-ts-{image_id}", iteration, summary_writer)
    return psnr.item()


def test_model(renderer, ts_dset, log_dir, batch_size, render_fn, plot_type="imageio", num_test_imgs=None):
    renderer.cuda()
    renderer.eval()
    if num_test_imgs is None:
        num_test_imgs = len(ts_dset)
    psnrs = []
    plot_every = max(1, num_test_imgs // 5)
    for image_id in tqdm(range(num_test_imgs), desc=f"test-dataset evaluation"):
        c_log_dir = log_dir if image_id % plot_every == plot_every - 1 else None
        psnr = plot_ts(ts_dset, 0, renderer, c_log_dir, iteration="test",
                       batch_size=batch_size, image_id=image_id, verbose=False,
                       plot_type=plot_type, render_fn=render_fn)
        psnrs.append(psnr)
    print(f"Average PSNR (over {num_test_imgs} poses): {np.mean(psnrs):.2f}")


def user_ask_options(prompt: str, opt1: str, opt2: str) -> str:
    prompt_wopt = f"{prompt} ({opt1}, {opt2})"
    while True:
        out = input(prompt_wopt)
        if out == opt1:
            return opt1
        elif out == opt2:
            return opt2
        else:
            print(f"Invalid option {out}. Please type '{opt1}' or '{opt2}'")


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    grid_id: int,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    timestamps: Optional[torch.Tensor] = None,
    # rendering options
    aabb: Optional[torch.Tensor] = None,
    near_plane: Optional[torch.Tensor] = None,
    far_plane: Optional[torch.Tensor] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thresh: float = 0.0,
    early_stop_eps: float = 1e-4,
    device="cuda:0",
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays_o.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays_o = rays_o.reshape([num_rays] + list(rays_o.shape[2:]))
        rays_d = rays_d.reshape([num_rays] + list(rays_d.shape[2:]))
    else:
        num_rays, _ = rays_shape
    if timestamps is not None:
        timestamps = timestamps.to(device)

    def sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays_o[ray_indices]
        t_dirs = chunk_rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        #print(f"sigma-fn with {len(ray_indices)} points")
        if timestamps is not None:
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, 0])
            )
            return radiance_field.query_density(positions, t)
        return radiance_field.query_density(positions, grid_id)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays_o[ray_indices]
        t_dirs = chunk_rays_d[ray_indices]
        #print(f"rgb-sigma-fn with {len(ray_indices)} points")
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, 0])
            )
            return radiance_field(positions, t_dirs, t)
        return radiance_field(positions, t_dirs, grid_id)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays_o = rays_o[i: i + chunk].to(device=device)
        chunk_rays_d = rays_d[i: i + chunk].to(device=device)
        chunk_rays_d = chunk_rays_d / torch.linalg.norm(chunk_rays_d, dim=-1, keepdim=True)

        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays_o,
            chunk_rays_d,
            scene_aabb=aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane.to(device=device) if near_plane is not None else None,
            far_plane=far_plane.to(device=device) if far_plane is not None else None,
            render_step_size=render_step_size,
            stratified=radiance_field.training,  # add random perturbations
            cone_angle=cone_angle,
            alpha_thre=alpha_thresh,
            early_stop_eps=early_stop_eps,
        )
        rgb, opacity, depth = rendering(
            t_starts=t_starts,
            t_ends=t_ends,
            ray_indices=ray_indices,
            n_rays=chunk_rays_o.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd.to(device=device) if render_bkgd is not None else None,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts), ray_indices, t_starts, t_ends, chunk_rays_o.shape[0]]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples, ray_indices, t_starts, t_ends, n_rays = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
        ray_indices,
        t_starts,
        t_ends,
        sum(n_rays),
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.999,
    last_epoch: int = -1
):
    """
    https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/optimization.py#L129
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(1e-5, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_step_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    milestones,
    gamma: float,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        out = 1.0
        for m in milestones:
            if current_step < m:
                break
            out *= gamma
        return out
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def init_dloader_random(worker_id):
    seed = torch.utils.data.get_worker_info().seed
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
