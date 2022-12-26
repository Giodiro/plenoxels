import os.path
import re
import subprocess
import tempfile
from typing import List

import numpy as np
import torch

from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import read_mp4, write_video_to_file, write_png


def eval_jod(pred_frames: List[np.ndarray], gt_frames: List[np.ndarray]) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_pred = os.path.join(tmpdir, "pred.mp4")
        write_video_to_file(file_pred, pred_frames)
        file_gt = os.path.join(tmpdir, "gt.mp4")
        write_video_to_file(file_gt, gt_frames)
        result = subprocess.check_output(['fvvdp', '--test', file_pred, '--ref', file_gt, '--gpu', '0', '--display', 'standard_fhd'])
        result = result.decode().strip()
        result = float(result.split('=')[1])
        print("JOD: ", result)
    return result


def extract_from_result(text: str, prompt: str):
    m = re.search(prompt, text)
    return float(m.group(1))


def eval_flip(pred_frames: List[np.ndarray], gt_frames: List[np.ndarray], interval=10) -> float:
    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_fname = os.path.join(tmpdir, "pred.png")
        gt_fname = os.path.join(tmpdir, "gt.png")
        for i in range(len(pred_frames)):
            if (i % interval) != 0:
                continue
            write_png(pred_fname, pred_frames[i])
            write_png(gt_fname, gt_frames[i])
            result = subprocess.check_output(['python', 'eval/flip.py', '--reference', gt_fname, '--test', pred_fname])
            result = result.decode()
            all_results.append({
                'Mean': extract_from_result(result, r'Mean: (\d+\.\d+)'),
                'Weighted median': extract_from_result(result, r'Weighted median: (\d+\.\d+)'),
                '1st weighted quartile': extract_from_result(result, r'1st weighted quartile: (\d+\.\d+)'),
                '3rd weighted quartile': extract_from_result(result, r'3rd weighted quartile: (\d+\.\d+)'),
                'Min': extract_from_result(result, r'Min: (\d+\.\d+)'),
                'Max': extract_from_result(result, r'Max: (\d+\.\d+)'),
            })
            print(all_results[-1])
    all_results_processed = {k: [_[k] for _ in all_results] for k in all_results[0]}
    # print(all_results_processed)
    all_results_processed = {k: sum(all_results_processed[k]) / len(all_results_processed[k]) for k in all_results_processed}
    print(all_results_processed)
    return all_results_processed['Mean']


def eval_metrics(video_path):
    frames = read_mp4(video_path)
    h1 = frames[0].shape[0] // 3
    pred_frames = [f[:h1].numpy() for f in frames]
    gt_frames = [f[h1:2*h1].numpy() for f in frames]

    psnrs, ssims, msssims, lpipss = [], [], [], []
    for pred, gt in zip(pred_frames, gt_frames):
        pred = torch.from_numpy(pred).float().div(255)
        gt = torch.from_numpy(pred).float().div(255)
        psnrs.append(metrics.psnr(pred, gt))
        ssims.append(metrics.ssim(pred, gt))
        msssims.append(metrics.msssim(pred, gt))
        lpipss.append(metrics.rgb_lpips(pred, gt, ))
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    msssim = np.mean(msssims)
    lpips = np.mean(lpipss)

    flip = eval_flip(pred_frames=pred_frames, gt_frames=gt_frames, interval=10)
    jod = eval_jod(pred_frames=pred_frames, gt_frames=gt_frames)

    print(f"Video at {video_path} metrics:")
    print(f"PSNR={psnr}")
    print(f"SSIM={ssim}")
    print(f"MS-SSIM={msssim}")
    print(f"Alex-LPIPS={lpips}")
    print(f"FLIP={flip}")
    print(f"JOD={jod}")


if __name__ == "__main__":
    eval_metrics("logs/flame_salmon/coffee_martini_ndc_f16_90k_ts1e-3_tspn1e-5_ptv2e-4_ptvpn2e-4_l1ap1e-4_l1appn1e-4_dl1e-3_v4bbox_150tdensity_150t-8d_coo+0.5/step90000.mp4")
