import numpy as np
import torch

from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import read_mp4


def eval_metrics(video_path):
    frames = read_mp4(video_path)
    h1 = frames[0].shape[0] // 3
    pred_frames = [f[:h1].numpy() for f in frames]
    gt_frames = [f[h1:2*h1].numpy() for f in frames]

    psnrs, ssims, msssims, lpipss = [], [], [], []
    for pred, gt in zip(pred_frames, gt_frames):
        pred = torch.from_numpy(pred).float().div(255)
        gt = torch.from_numpy(gt).float().div(255)
        psnrs.append(metrics.psnr(pred, gt))
        ssims.append(metrics.ssim(pred, gt))
        msssims.append(metrics.msssim(pred, gt))
        lpipss.append(metrics.rgb_lpips(pred, gt, net_name="alex"))
    psnr = np.mean(psnrs)
    ssim = np.mean(ssims)
    msssim = np.mean(msssims)
    lpips = np.mean(lpipss)

    flip = metrics.flip(pred_frames=pred_frames, gt_frames=gt_frames, interval=10)
    jod = metrics.jod(pred_frames=pred_frames, gt_frames=gt_frames)

    print()
    print(f"Video at {video_path} metrics:")
    print(f"PSNR={psnr}")
    print(f"SSIM={ssim}")
    print(f"MS-SSIM={msssim}")
    print(f"Alex-LPIPS={lpips}")
    print(f"FLIP={flip}")
    print(f"JOD={jod}")
    print()
    print()


if __name__ == "__main__":
    #eval_metrics("logs/flame_salmon/coffee_martini_ndc_f16_90k_ts1e-3_tspn1e-5_ptv2e-4_ptvpn2e-4_l1ap1e-4_l1appn1e-4_dl1e-3_v4bbox_150tdensity_150t-8d_coo+0.5/step90000.mp4")
    #eval_metrics("logs/flame_salmon/flame_steak_ndc_f16_90k_ts1e-3_tspn1e-5_ptv2e-4_ptvpn2e-4_l1ap1e-4_l1appn1e-4_dl1e-3_v4bbox_150tdensity_150t-8d_coo+0.5/step90000.mp4")
    #eval_metrics("logs/flame_salmon/sear_steak_ndc_f16_90k_ts1e-3_tspn1e-5_ptv2e-4_ptvpn2e-4_l1ap1e-4_l1appn1e-4_dl1e-3_v4bbox_150tdensity_150t-8d_coo+0.5/step90000.mp4")
    eval_metrics("logs/flame_salmon/cook_spinach_ndc_f16_90k_ts1e-3_tspn1e-5_ptv2e-4_ptvpn2e-4_l1ap1e-4_l1appn1e-4_dl1e-3_v4bbox_150tdensity_150t-8d_coo+0.5/step90000.mp4")
    eval_metrics("logs/flame_salmon/cut_roasted_beef_ndc_f16_90k_ts1e-3_tspn1e-5_ptv2e-4_ptvpn2e-4_l1ap1e-4_l1appn1e-4_dl1e-3_v4bbox_150tdensity_150t-8d_coo+0.5_v2/step90000.mp4")
    eval_metrics("logs/flame_salmon/flame_salmon_ndc_f16_90k_ts1e-3_tspn1e-5_ptv2e-4_ptvpn2e-4_l1ap1e-4_l1appn1e-4_dl1e-3_v4bbox_150tdensity_150t-8d_coo+0.5/step90000.mp4")
