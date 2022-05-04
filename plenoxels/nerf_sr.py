import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from plenoxels.corner_tree import CornerTree
from plenoxels.fsrcnn import FSRCNN
from plenoxels.swin_ir import SwinIR
from plenoxels.synthetic_nerf_dataset import MultiSyntheticNerfDataset, MultiSyntheticNerfDatasetv2
from plenoxels.tc_plenoptimize import parse_config


def init_multi_dset(cfg):
    assert isinstance(cfg.data.datadir, list), "Need to provide multiple data-directories"

    tr_dset = MultiSyntheticNerfDataset(
        cfg.data.datadir, split='train', low_resolution=cfg.multi_sr.low_resolution,
        high_resolution=cfg.multi_sr.high_resolution, max_frames=cfg.data.max_tr_frames)
    ts_dset = MultiSyntheticNerfDataset(
        cfg.data.test_datadir, split='test', low_resolution=cfg.multi_sr.low_resolution,
        high_resolution=cfg.multi_sr.high_resolution, max_frames=cfg.data.max_ts_frames)

    return tr_dset, ts_dset


def init_multi_dset_v2(cfg):
    assert isinstance(cfg.data.datadir, list), "Need to provide multiple data-directories"

    tr_dset = MultiSyntheticNerfDatasetv2(
        cfg.data.datadir, split='train', low_resolution=cfg.multi_sr.low_resolution,
        high_resolution=cfg.multi_sr.high_resolution, max_frames=cfg.data.max_tr_frames,
        patch_size=cfg.multi_sr.patch_size)
    ts_dset = MultiSyntheticNerfDatasetv2(
        cfg.data.datadir, split='test', low_resolution=cfg.multi_sr.low_resolution,
        high_resolution=cfg.multi_sr.high_resolution, max_frames=cfg.data.max_ts_frames,
        patch_size=cfg.multi_sr.patch_size)
    tr_loader = DataLoader(tr_dset, batch_size=cfg.multi_sr.batch_size, pin_memory=True,
                           drop_last=False, shuffle=True)
    ts_loader = DataLoader(ts_dset, batch_size=cfg.multi_sr.batch_size, pin_memory=True,
                           drop_last=False, shuffle=True)

    return tr_dset, ts_dset, tr_loader, ts_loader


def init_plenoxels(cfg, tr_dset):
    tree = CornerTree(sh_degree=cfg.sh.degree, init_internal=299593,
                      aabb=tr_dset.scene_bbox, near=2., far=6.,
                      init_rgb=0.01, init_sigma=0.1, max_samples_per_node=cfg.corner_tree.max_samples_per_node,
                      max_intersections=cfg.corner_tree.max_intersections, sigma_thresh=cfg.corner_tree.sigma_thresh,
                      stop_thresh=cfg.corner_tree.stop_thresh)
    for i in range(cfg.multi_sr.tree_height):
        tree.refine()
    return tree


def init_sr(cfg):
    upscale = cfg.multi_sr.high_resolution / cfg.multi_sr.low_resolution
    assert abs(upscale - int(upscale)) < 1e-9
    if cfg.multi_sr.sr_model.lower() == "fsrcnn":
        sr = FSRCNN(upscale_factor=int(upscale))
    elif cfg.multi_sr.sr_model.lower() == "swin-ir":
        sr = SwinIR(upscale=int(upscale),
                    in_chans=3,
                    img_size=(cfg.multi_sr.patch_size, cfg.multi_sr.patch_size),
                    window_size=8,                     # always same from paper
                    img_range=1.,                      # we use float images
                    depths=[6, 6, 6, 6],               # from paper (lightweight config)
                    embed_dim=60,                      # from paper (lightweight config)
                    num_heads=[6, 6, 6, 6],            # from paper (lightweight config)
                    mlp_ratio=2,                       # from paper (lightweight config)
                    upsampler="pixelshuffledirect",    # from paper (lightweight config)
                    resi_connection="1conv",           # from paper (lightweight config)
                    )
    else:
        raise RuntimeError("model type not understood")
    return sr


def run_tree_epoch(plenoxel_list, loader, loss_fn, optim):
    dev = "cuda"
    psnr, mse = [], []
    e_start = time.time()

    for i, data in enumerate(loader):
        num_scenes = len(data['scene_id'])
        lr = data['low'].view(num_scenes, -1, 3).to(dev)  # N*H*W, 3

        rays_o = data['rays'][:, :, :, 0, :].view(num_scenes, -1, 3).contiguous().to(dev)
        rays_d = data['rays'][:, :, :, 1, :].view(num_scenes, -1, 3).contiguous().to(dev)
        for j, s_id in enumerate(data['scene_id']):
            pred_lr = plenoxel_list[s_id](rays_o[j], rays_d[j], use_ext=True)
            loss = loss_fn(pred_lr, lr[j])
            if optim is not None:
                optim_ = optim
                if isinstance(optim, list):
                    optim_ = optim[s_id]
                optim_.zero_grad()
                loss.backward()
                optim_.step()

            with torch.no_grad():
                mse.append(loss.item())
                psnr.append(-10.0 * math.log(mse[-1]) / math.log(10.0))

    return {
        "mse": np.mean(mse),
        "psnr": np.mean(psnr),
        "time": time.time() - e_start,
    }


def run_sr_epoch(plenoxel_list, sr, loader, loss_fn, optim, grad_scaler):
    dev = "cuda"
    psnr, mse = [], []
    e_start = time.time()

    for i, data in enumerate(loader):
        num_scenes = len(data['scene_id'])
        l_patch_size = data['low'].shape[1]

        hr = data['high'].permute(0, 3, 1, 2).to(dev)  # N, 3, nH, nW

        rays_o = data['rays'][:, :, :, 0, :].view(num_scenes, -1, 3).contiguous().to(dev)
        rays_d = data['rays'][:, :, :, 1, :].view(num_scenes, -1, 3).contiguous().to(dev)
        with torch.no_grad():
            pred_lr = []
            for j, s_id in enumerate(data['scene_id']):
                pred_lr.append(
                    plenoxel_list[s_id](rays_o[j], rays_d[j], use_ext=True)
                        .reshape(l_patch_size, l_patch_size, 3).permute(2, 0, 1)
                )
            pred_lr = torch.stack(pred_lr, 0)  # N, 3, H, W
        pred_hr = sr(pred_lr)  # N, 3, nH, nW
        loss = loss_fn(pred_hr, hr)

        if optim is not None:
            optim.zero_grad()
            if grad_scaler is None:
                loss.backward()
                optim.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optim)
                grad_scaler.update()

        with torch.no_grad():
            mse.append(loss.item())
            psnr.append(-10.0 * math.log(mse[-1]) / math.log(10.0))

    return {
        "mse": np.mean(mse),
        "psnr": np.mean(psnr),
        "time": time.time() - e_start,
    }


def run_epoch(plenoxel_list, super_res, dset, loss_fn, optim_list, grad_scaler):
    dev = "cuda"
    psnr, mse, lr_mse = [], [], []
    e_start = time.time()
    data_ids = np.random.permutation(len(dset))
    for did in data_ids:
        rays, pxls_lr, pxls_hr, scene_id = dset[did]
        scene_id = scene_id.item()
        #print("Scene: %d" % (scene_id))
        rays_o = rays[:, 0].contiguous().to(dev)
        rays_d = rays[:, 1].contiguous().to(dev)
        rays_d.div_(torch.linalg.norm(rays_d, dim=1, keepdim=True))
        pxls_hr = pxls_hr.to(dev)
        pxls_lr = pxls_lr.to(dev)
        plenoxel = plenoxel_list[scene_id]
        #optim = optim_list[scene_id] if optim_list is not None else None
        optim = optim_list

        with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
            pred_pxls_lr = plenoxel(rays_o, rays_d, use_ext=True)
            pred_pxls_lr_cnn = pred_pxls_lr.view(1, dset.low_resolution, dset.low_resolution, 3).permute(0, 3, 1, 2).contiguous()
            pred_pxls_hr = super_res(pred_pxls_lr_cnn).view(3, -1).T
            loss = loss_fn(pred_pxls_hr, pxls_hr) + loss_fn(pred_pxls_lr, pxls_lr)

        if optim is not None:
            if grad_scaler is None:
                loss.backward()
                optim.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optim)
                grad_scaler.update()
            optim.zero_grad()

        with torch.no_grad():
            psnr.append(-10.0 * math.log(loss.item()) / math.log(10.0))
            mse.append(loss.item())
            lr_mse.append(loss_fn(pred_pxls_lr, pxls_lr).item())

    return {
        "mse": np.mean(mse),  # Final MSE
        "psnr": np.mean(psnr),  # Final PSNR
        "lr_mse": np.mean(lr_mse),  # MSE of the low-resolution image (only plenoxels)
        "time": time.time() - e_start,
    }


def train(cfg):
    tr_dset, ts_dset = init_multi_dset(cfg)
    plenoxels_list = [init_plenoxels(cfg, tr_dset).cuda() for _ in range(len(cfg.data.datadir))]
    sr = init_sr(cfg).cuda()

    loss = torch.nn.MSELoss()

    params = [
        {"params": sr.feature_extraction.parameters()},
        {"params": sr.shrink.parameters()},
        {"params": sr.map.parameters()},
        {"params": sr.expand.parameters()},
        {"params": sr.deconv.parameters(), "lr": cfg.multi_sr.sr_lr * 0.1},
    ]
    for plenoxels in plenoxels_list:
        params.append({"params": plenoxels.parameters(), "lr": cfg.optim.lr})

    optimizer_list = torch.optim.SGD(params,
        lr=cfg.multi_sr.sr_lr,
        momentum=cfg.multi_sr.momentum,
        weight_decay=cfg.multi_sr.weight_decay
        )
    scaler = None
    if cfg.multi_sr.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Training loop
    rs = {
        "train": {
            "mse": [],
            "lr_mse": [],
            "psnr": [],
            "time": [],
        },
        "test": {
            "mse": [],
            "lr_mse": [],
            "psnr": [],
            "time": [],
        },
    }
    for epoch in range(cfg.optim.num_epochs):
        [plenoxels.train() for plenoxels in plenoxels_list]
        sr.train()
        tr_rs = run_epoch(plenoxels_list, sr, tr_dset, loss, optim_list=optimizer_list, grad_scaler=scaler)

        [plenoxels.eval() for plenoxels in plenoxels_list]
        sr.eval()
        with torch.no_grad():
            ts_rs = run_epoch(plenoxels_list, sr, ts_dset, loss, optim_list=None, grad_scaler=scaler)

        rs["train"]["mse"].append(tr_rs["mse"])
        rs["train"]["psnr"].append(tr_rs["psnr"])
        rs["train"]["lr_mse"].append(tr_rs["lr_mse"])
        rs["train"]["time"].append(tr_rs["time"])
        rs["test"]["mse"].append(ts_rs["mse"])
        rs["test"]["psnr"].append(ts_rs["psnr"])
        rs["test"]["lr_mse"].append(ts_rs["lr_mse"])
        rs["test"]["time"].append(ts_rs["time"])
        print(f"Epoch {epoch:4d} "
              f"- train({tr_rs['time']:.2f}s) MSE={tr_rs['mse']:.4f}  LR-MSE={tr_rs['lr_mse']:.4f}  PSNR {tr_rs['psnr']:.2f} "
              f"- test ({ts_rs['time']:.2f}s) MSE={ts_rs['mse']:.4f}  LR-MSE={ts_rs['lr_mse']:.4f}  PSNR {ts_rs['psnr']:.2f}")


if __name__ == "__main__":
    _cfg = parse_config()
    train(_cfg)
