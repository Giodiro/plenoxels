import os.path
import subprocess
import tempfile

from plenoxels.ops.image.io import read_mp4, write_video_to_file


def eval_jod(video_path: str) -> float:
    frames = read_mp4(video_path)
    pred_frames = [f[:1024].numpy() for f in frames]
    gt_frames = [f[1024:2048].numpy() for f in frames]

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


def eval_metrics(video_path):
    eval_jod(video_path=video_path)
