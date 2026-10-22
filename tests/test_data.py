"""Sanity checks on the raw cars data. Run from anywhere: python tests/test_data.py"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from SRNDataset import txt_to_matrix, EXCLUDED_OBJECTS

ROOT = Path(__file__).resolve().parent.parent / "cars_train"
TOL = 1e-5   # poses are float32 written as text, measured errors are ~1e-6

def objects():
    return sorted(p for p in ROOT.glob("[!.]*") if p.is_dir() and p.name not in EXCLUDED_OBJECTS)   # same rule as DLDataset

@lru_cache(maxsize=None)
def all_poses():
    # [N, 4, 4] pose of every view of every object, and matching "object/view" labels for error messages
    poses, labels = [], []
    for obj in objects():
        for txt in sorted((obj / "pose").glob("*.txt")):
            poses.append(txt_to_matrix(txt))
            labels.append(f"{obj.name}/{txt.stem}")
    return np.stack(poses), labels

def test_pose_and_rgb_names_match():
    # DLDataset lists views from pose/ and then opens rgb/<view>.png, so the two must agree
    mismatched = []
    for obj in objects():
        pose_names = sorted(txt.stem for txt in (obj / "pose").glob("*.txt"))
        rgb_names = sorted(png.stem for png in (obj / "rgb").glob("*.png"))
        if pose_names != rgb_names:
            diff = sorted(set(pose_names) ^ set(rgb_names))
            mismatched.append(f"{obj.name}: {len(pose_names)} poses, {len(rgb_names)} rgbs, differing e.g. {diff[:3]}")
    assert not mismatched, f"{len(mismatched)}/{len(objects())} objects have different pose and rgb names:\n" + "\n".join(mismatched)

def test_radius_is_same_for_all_views():
    poses, labels = all_poses()
    radius = np.linalg.norm(poses[:, :3, 3], axis=1)
    lo, hi = radius.argmin(), radius.argmax()
    assert radius[hi] - radius[lo] < TOL, f"radius goes from {radius[lo]} ({labels[lo]}) to {radius[hi]} ({labels[hi]})"

def test_rotation_block_is_proper_rotation():
    poses, labels = all_poses()
    R = poses[:, :3, :3]
    orth_err = np.abs(R.transpose(0, 2, 1) @ R - np.eye(3)).max(axis=(1, 2))   # R^T R = I
    worst = orth_err.argmax()
    assert orth_err[worst] < TOL, f"R^T R differs from I by {orth_err[worst]} ({labels[worst]})"
    det_err = np.abs(np.linalg.det(R) - 1)   # det = +1 rules out reflections
    worst = det_err.argmax()
    assert det_err[worst] < TOL, f"det(R) differs from 1 by {det_err[worst]} ({labels[worst]})"

def test_camera_forward_axis_points_to_origin():
    poses, labels = all_poses()
    forward = poses[:, :3, 2]   # R[:, 2]
    t = poses[:, :3, 3]
    to_origin = -t / np.linalg.norm(t, axis=1, keepdims=True)
    cos = (forward * to_origin).sum(axis=1)
    worst = cos.argmin()
    assert 1 - cos[worst] < TOL, f"R[:, 2] and the direction to the origin have cosine {cos[worst]} ({labels[worst]})"

if __name__ == "__main__":
    tests = [fn for name, fn in dict(globals()).items() if name.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL", fn.__name__)
            print(e)
    sys.exit(1 if failed else 0)
