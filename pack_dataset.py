"""Pack one split (cars_train / cars_val / cars_test) into a few big arrays. See docs/DATA.md.

    python pack_dataset.py --split train --out packed

writes <out>/<split>_images.npy [objects, views, 64, 64, 3] uint8, <out>/<split>_directions.npy [objects, views, 3] float32
and <out>/<split>_manifest.json. Any problem with the data stops the run and names the object.
"""
import argparse
import json
import multiprocessing
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from SRNDataset import EXCLUDED_OBJECTS, txt_to_matrix, get_cam_unit_vector

SRC_SIZE = 128
OUT_SIZE = 64
BACKGROUND = (255, 255, 255)
RADIUS = 1.3
RADIUS_TOL = 1e-5      # measured spread over the train split is ~6e-7 (float32 poses)
UNIT_TOL = 1e-6
RESIZE = "2x2 box average of the 0-255 values after compositing, rounded half up to uint8"

def check(condition, message):
    # not assert, so the checks survive python -O
    if not condition:
        raise ValueError(message)

def pack_image(png):
    im = Image.open(png)
    check(im.mode == "RGBA", f"{png.name}: mode {im.mode}, expected RGBA")
    check(im.size == (SRC_SIZE, SRC_SIZE), f"{png.name}: size {im.size}, expected {(SRC_SIZE, SRC_SIZE)}")
    alpha = np.asarray(im)[..., 3]
    check(alpha.min() == 255, f"{png.name}: alpha goes down to {alpha.min()}, expected 255 everywhere")
    bg = Image.new("RGBA", im.size, BACKGROUND + (255,))
    rgb = np.asarray(Image.alpha_composite(bg, im).convert("RGB"))
    block_sum = rgb.astype(np.uint16).reshape(OUT_SIZE, 2, OUT_SIZE, 2, 3).sum(axis=(1, 3))
    return ((block_sum + 2) >> 2).astype(np.uint8)   # (sum / 4) rounded half up, in integers

def pack_direction(txt):
    pose = txt_to_matrix(txt)
    radius = np.linalg.norm(pose[:3, 3])
    check(abs(radius - RADIUS) < RADIUS_TOL, f"{txt.name}: camera radius {radius}, expected {RADIUS}")
    direction = get_cam_unit_vector(pose).astype(np.float32)
    norm = np.linalg.norm(direction.astype(np.float64))
    check(abs(norm - 1) < UNIT_TOL, f"{txt.name}: direction length {norm}, expected 1")
    return direction

def process_object(task):
    # runs in a worker: decodes every view of one object and returns it with its slot
    index, obj, views = task
    images = np.empty((len(views), OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
    directions = np.empty((len(views), 3), dtype=np.float32)
    view = None
    try:
        for k, view in enumerate(views):
            images[k] = pack_image(obj / "rgb" / f"{view}.png")
            directions[k] = pack_direction(obj / "pose" / f"{view}.txt")
    except Exception as e:
        raise RuntimeError(f"object {obj.name}, view {view}: {type(e).__name__}: {e}") from None
    return index, images, directions

def list_views(root, limit):
    objects = sorted(p for p in root.glob("[!.]*") if p.is_dir() and p.name not in EXCLUDED_OBJECTS)
    check(objects, f"no objects found in {root}")
    objects = objects[:limit]
    views = []
    for obj in objects:
        pose_names = sorted(txt.stem for txt in (obj / "pose").glob("*.txt"))
        rgb_names = sorted(png.stem for png in (obj / "rgb").glob("*.png"))
        diff = sorted(set(pose_names) ^ set(rgb_names))
        check(pose_names == rgb_names, f"object {obj.name}: {len(pose_names)} poses but {len(rgb_names)} rgbs, differing e.g. {diff[:3]}")
        views.append(pose_names)
    return objects, views

def write_manifest(path, root, split, objects, views, n_views, excluded_present, limit):
    shared = all(v == views[0] for v in views)
    manifest = {
        "split": split,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(root),
        "limit": limit,
        "n_objects": len(objects),
        "n_views_per_object": n_views,
        "n_images": len(objects) * n_views,
        "images": {"file": f"{split}_images.npy", "shape": [len(objects), n_views, OUT_SIZE, OUT_SIZE, 3], "dtype": "uint8"},
        "directions": {"file": f"{split}_directions.npy", "shape": [len(objects), n_views, 3], "dtype": "float32"},
        "resize": RESIZE,
        "background": list(BACKGROUND),
        "radius": RADIUS,
        "radius_tolerance": RADIUS_TOL,
        "exclusion_list": sorted(EXCLUDED_OBJECTS),
        "excluded_present_in_split": excluded_present,
        "object_ids": [obj.name for obj in objects],
    }
    if shared:
        manifest["view_names"] = views[0]   # slot k of every object is this view
    else:
        manifest["view_names_per_object"] = {obj.name: v for obj, v in zip(objects, views)}
    path.write_text(json.dumps(manifest, indent=1) + "\n")

def pack(root, out, split, workers=1, limit=None, expect_views=None):
    root, out = Path(root), Path(out)
    objects, views = list_views(root, limit)
    n_views = len(views[0])
    for obj, v in zip(objects, views):
        check(len(v) == n_views, f"object {obj.name} has {len(v)} views, but {objects[0].name} has {n_views}: the split is ragged")
    check(expect_views is None or n_views == expect_views, f"found {n_views} views per object, expected {expect_views}")
    excluded_present = sorted(name for name in EXCLUDED_OBJECTS if (root / name).is_dir())

    out.mkdir(parents=True, exist_ok=True)
    final = {k: out / f"{split}_{k}.npy" for k in ("images", "directions")}
    final["manifest"] = out / f"{split}_manifest.json"
    tmp = {k: p.with_name(p.name + ".tmp") for k, p in final.items()}

    try:
        # output file is created on disk and filled object by object: RAM stays at a few objects
        images = np.lib.format.open_memmap(tmp["images"], mode="w+", dtype=np.uint8, shape=(len(objects), n_views, OUT_SIZE, OUT_SIZE, 3))
        directions = np.zeros((len(objects), n_views, 3), dtype=np.float32)
        filled = np.zeros(len(objects), dtype=bool)

        tasks = [(i, obj, v) for i, (obj, v) in enumerate(zip(objects, views))]
        if workers > 1:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                results = pool.imap_unordered(process_object, tasks)
                for i, im, d in tqdm(results, total=len(tasks), desc=split):
                    images[i], directions[i], filled[i] = im, d, True   # slot fixed by sorted order, not by finish order
        else:
            for i, im, d in tqdm(map(process_object, tasks), total=len(tasks), desc=split):
                images[i], directions[i], filled[i] = im, d, True
        check(filled.all(), f"objects never written: {[objects[i].name for i in np.flatnonzero(~filled)]}")

        images.flush()
        del images
        with open(tmp["directions"], "wb") as f:
            np.save(f, directions)
        write_manifest(tmp["manifest"], root, split, objects, views, n_views, excluded_present, limit)
        # rename at the very end, manifest last: a manifest next to the arrays means the run finished
        os.replace(tmp["images"], final["images"])
        os.replace(tmp["directions"], final["directions"])
        os.replace(tmp["manifest"], final["manifest"])
    except BaseException:
        for p in tmp.values():
            p.unlink(missing_ok=True)
        raise
    return final

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default=str(here), help="folder holding cars_<split>/")
    parser.add_argument("--out", default=str(here / "packed"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None, help="only the first N objects (sorted), for quick runs")
    parser.add_argument("--views", type=int, default=None, help="fail unless every object has exactly this many views")
    args = parser.parse_args()

    files = pack(Path(args.data_dir) / f"cars_{args.split}", args.out, args.split, args.workers, args.limit, args.views)
    for name, path in files.items():
        print(f"{name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
