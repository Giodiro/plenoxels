import os.path
import re
import subprocess
import tempfile

from plenoxels.ops.image.io import read_mp4, write_video_to_file, write_png


def eval_jod(video_path: str) -> float:
    frames = read_mp4(video_path)
    h1 = frames[0].shape[0] // 3
    pred_frames = [f[:h1].numpy() for f in frames]
    gt_frames = [f[h1:2*h1].numpy() for f in frames]

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


def eval_flip(video_path, interval=10):
    # Video to png
    frames = read_mp4(video_path)
    h1 = frames[0].shape[0] // 3
    pred_frames = [f[:h1].numpy() for f in frames]
    gt_frames = [f[h1:2*h1].numpy() for f in frames]
    all_results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_fname = os.path.join(tmpdir, "pred.png")
        gt_fname = os.path.join(tmpdir, "gt.png")
        for i in range(len(frames)):
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


def eval_metrics(video_path):
    eval_jod(video_path=video_path)
