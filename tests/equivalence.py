"""Packed arrays vs the original loader: both must describe the same images and directions in the same order.
Needs packed/train_* (python pack_dataset.py --split train) and cars_train/. Run from anywhere: python tests/equivalence.py"""
import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from SNRDataset import DLDataset, PackedDataset

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE / "cars_train"
PACKED = HERE / "packed"
SPLIT = "train"
IMG_TOL = 1 / 127.5 + 1e-6   # one 8-bit rounding step in the [-1, 1] range (~0.008)
DIR_TOL = 1e-6               # both come from the same pose files, so essentially identical
N_RANDOM_OBJECTS = 18        # plus the first and last object
OBSERVED = {"image": 0.0, "direction": 0.0}   # largest differences seen, printed by the runner

@lru_cache(maxsize=None)
def packed():
    manifest_path = PACKED / f"{SPLIT}_manifest.json"
    assert manifest_path.exists(), f"{manifest_path} not found: run python pack_dataset.py --split {SPLIT}"
    manifest = json.loads(manifest_path.read_text())
    images = np.load(PACKED / manifest["images"]["file"], mmap_mode="r")
    directions = np.load(PACKED / manifest["directions"]["file"])
    return manifest, images, directions

@lru_cache(maxsize=None)
def loader():
    return DLDataset(ROOT)

def sample_objects(n_objects):
    # first and last object (off-by-one on the object axis) plus a seeded random sample
    rest = random.Random(0).sample(range(1, n_objects - 1), N_RANDOM_OBJECTS)
    return sorted({0, n_objects - 1, *rest})

def test_object_order_is_sorted_folder_order():
    manifest, images, directions = packed()
    exclude = set(manifest["exclusion_list"])
    folders = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in exclude)
    folders = folders[:manifest["limit"]]
    assert manifest["object_ids"] == folders, "manifest object order differs from the sorted folders"
    assert manifest["object_ids"] == [obj.name for obj in loader().objects][:len(folders)], "manifest order differs from the loader's order"
    assert images.shape[0] == directions.shape[0] == len(folders) == manifest["n_objects"]
    assert not exclude & set(manifest["object_ids"])

def test_view_k_is_view_k_in_the_file_names():
    manifest, images, directions = packed()
    n_views = images.shape[1]
    assert directions.shape[1] == n_views == manifest["n_views_per_object"] == len(manifest["view_names"])
    ds = loader()
    names = {obj.name: i for i, obj in enumerate(ds.objects)}
    for o in sample_objects(len(manifest["object_ids"])):
        name = manifest["object_ids"][o]
        obj = ROOT / name
        pose_names = sorted(txt.stem for txt in (obj / "pose").glob("*.txt"))
        rgb_names = sorted(png.stem for png in (obj / "rgb").glob("*.png"))
        assert pose_names == rgb_names == manifest["view_names"], f"{name}: file names differ from the manifest's view names"
        assert ds.views[names[name]] == manifest["view_names"], f"{name}: loader views differ from the manifest's view names"
    assert manifest["view_names"] == sorted(manifest["view_names"])

def check_pairs(manifest, images, directions, objects, rng):
    # for each sampled object, ask the loader for a pair made of two chosen views (first and last, then two random ones)
    # and compare both returned images and both directions with the packed slots of exactly those views
    ds = loader()
    names = {obj.name: i for i, obj in enumerate(ds.objects)}
    n_views = images.shape[1]
    for o in objects:
        name = manifest["object_ids"][o]
        idx = names[name]
        a, b = rng.sample(range(1, n_views - 1), 2)
        for k1, k2 in ((0, n_views - 1), (a, b)):
            saved = ds.views[idx]
            ds.views[idx] = [manifest["view_names"][k1], manifest["view_names"][k2]]   # the draw is then these two, in random order
            try:
                img1, img2, d_pose, obj_id = ds[idx]
            finally:
                ds.views[idx] = saved
            assert obj_id == name
            results = []
            for first, second in ((k1, k2), (k2, k1)):   # either of the two may have been drawn as source
                got = (img1.numpy(), img2.numpy(), d_pose[:3].numpy(), d_pose[3:].numpy())
                want = [images[o, k].astype(np.float32).transpose(2, 0, 1) / 127.5 - 1 for k in (first, second)] \
                     + [directions[o, first], directions[o, second]]
                err = [np.abs(g - w).max() for g, w in zip(got, want)]
                results.append((first, second, err))
            ok = [r for r in results if max(r[2][:2]) <= IMG_TOL and max(r[2][2:]) <= DIR_TOL]
            assert ok, (f"{name}, views {manifest['view_names'][k1]} & {manifest['view_names'][k2]}: loader and packed data disagree "
                        f"(max |diff| for src img, tgt img, src dir, tgt dir, per possible order: {[(r[0], r[1], [float(f'{e:.4g}') for e in r[2]]) for r in results]})")
            OBSERVED["image"] = max(OBSERVED["image"], *ok[0][2][:2])
            OBSERVED["direction"] = max(OBSERVED["direction"], *ok[0][2][2:])

def test_images_and_directions_match_the_loader():
    manifest, images, directions = packed()
    check_pairs(manifest, images, directions, sample_objects(len(manifest["object_ids"])), random.Random(1))

def test_packed_dataset_returns_same_items_as_raw_loader():
    # same seed -> both datasets draw the same two views (same view lists, same random.sample call), so items must agree
    manifest, _, _ = packed()
    raw, fast = loader(), PackedDataset(PACKED, SPLIT)
    assert len(raw) == len(fast) == manifest["n_objects"]
    for idx in sample_objects(len(raw)):
        for seed in range(3):
            random.seed(seed)
            a = raw[idx]
            random.seed(seed)
            b = fast[idx]
            assert a[3] == b[3] == manifest["object_ids"][idx]
            for x, y in zip(a[:3], b[:3]):
                assert x.shape == y.shape and x.dtype == y.dtype
            img_err = max((a[0] - b[0]).abs().max().item(), (a[1] - b[1]).abs().max().item())
            dir_err = (a[2] - b[2]).abs().max().item()
            assert img_err <= IMG_TOL and dir_err <= DIR_TOL, f"{a[3]} seed {seed}: image diff {img_err}, direction diff {dir_err}"
            OBSERVED["image"] = max(OBSERVED["image"], img_err)
            OBSERVED["direction"] = max(OBSERVED["direction"], dir_err)

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
    print(f"largest differences seen: image {OBSERVED['image']:.5f} (allowed {IMG_TOL:.5f}), direction {OBSERVED['direction']:.2e} (allowed {DIR_TOL:.0e})")
    sys.exit(1 if failed else 0)
