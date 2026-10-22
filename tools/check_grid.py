"""Save a picture of what the loader returns, to check the pairing by eye.

    python tools/check_grid.py [--rows 8] [--seed 0] [--out outputs/loader_grid.png]

One row per pair: source image, target image, and under each its camera direction (xyz and elevation / azimuth).
The target should look like the same car as the source, seen from another side. The last two rows are forced pole
pairs (top-most and bottom-most camera of an object). A reviewed copy is in docs/img/loader_grid.png.
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from SNRDataset import DLDataset, txt_to_matrix, get_cam_unit_vector

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=str(HERE / "cars_train"))
parser.add_argument("--rows", type=int, default=8, help="random pairs")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=str(HERE / "outputs" / "loader_grid.png"))
args = parser.parse_args()

random.seed(args.seed)
ds = DLDataset(args.root)

# (object index, restricted views or None); the last two rows force the camera to the highest / lowest views of an object
rows = [(random.randrange(len(ds)), None) for _ in range(args.rows)]
for pole in (+1, -1):
    idx = random.randrange(len(ds))
    obj = ds.objects[idx]
    z = {v: get_cam_unit_vector(txt_to_matrix(obj / "pose" / f"{v}.txt"))[2] for v in ds.views[idx]}
    top, bottom = max(z, key=z.get), min(z, key=z.get)
    rows.append((idx, [top, bottom] if pole > 0 else [bottom, top]))

def label(d):
    el, az = np.degrees(np.arcsin(d[2])), np.degrees(np.arctan2(d[1], d[0]))
    return f"({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})\nelev {el:+.0f}  azim {az:+.0f}"

fig, axes = plt.subplots(len(rows), 2, figsize=(5.6, 2.9 * len(rows)))
for r, (idx, views) in enumerate(rows):
    saved = ds.views[idx]
    if views:
        ds.views[idx] = views
    img1, img2, d_pose, obj_id = ds[idx]
    ds.views[idx] = saved
    for c, (img, d, role) in enumerate(((img1, d_pose[:3], "source"), (img2, d_pose[3:], "target"))):
        ax = axes[r, c]
        ax.imshow(((img + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{role} dir\n{label(d.numpy())}", fontsize=7)
        if c == 0:
            ax.set_ylabel(obj_id[:8] + ("  (poles)" if views else ""), fontsize=7)
fig.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=130)
print(f"saved {len(rows)} rows ({args.rows} random, 2 pole) to {args.out}")
