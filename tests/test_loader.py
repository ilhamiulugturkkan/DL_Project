"""Loader tests. Fixture tests build a tiny fake dataset (no real data needed, well under a second);
real-data tests use a small random sample of cars_train. Run from anywhere: python tests/test_loader.py"""
import random
import sys
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from SNRDataset import DLDataset, PackedDataset
from pack_dataset import pack

ROOT = Path(__file__).resolve().parent.parent / "cars_train"
PACKED = Path(__file__).resolve().parent.parent / "packed"

# which loader the tests run against: "raw" (DLDataset on the png/txt folders) or "packed" (PackedDataset on the arrays).
# The runner below goes through both; pytest would only see the default.
KIND = "raw"
# max |difference| allowed between a returned image and an independent re-read of the png: float math vs 8-bit rounding
IMG_TOL = {"raw": 1e-5, "packed": 0.5 / 127.5 + 1e-6}

# ---------- synthetic fixture: 2 objects x 5 views ----------
# Every view has a flat-color image encoding (object, view) and a pose with a known direction, so a test can
# decode which view the returned image came from and compare it with the view the returned pose came from.

OBJS = ["obj_a", "obj_b"]
VIEWS = [f"{v:06d}" for v in range(5)]
RADIUS = 1.3
S = 3 ** -0.5
DIRS = {   # known camera direction (unit vector) of every view; views 0 and 3 of obj_a sit exactly on the poles
    "obj_a": [(0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, -1), (S, S, S)],
    "obj_b": [(0, 1, 0), (0, 0, 1), (-1, 0, 0), (S, -S, S), (0, 0, -1)],
}

def color(o, v):
    return (20 + 40 * v, 100 + 100 * o, 50)   # R encodes the view, G the object

def decode(img):
    # solid-color [3, H, W] image in [-1, 1] -> (object index, view index)
    assert (img.amax(dim=(1, 2)) - img.amin(dim=(1, 2))).max().item() < 1e-4, "fixture images are flat per channel, this one is not"
    r, g, _ = ((img.mean(dim=(1, 2)) + 1) * 127.5).round().tolist()
    return round((g - 100) / 100), round((r - 20) / 40)

def pose_text(direction):
    # camera-to-world 4x4, one row-major line like the real files: camera at RADIUS * direction, looking at the origin
    d = np.array(direction, dtype=float)
    forward = -d
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    P = np.eye(4)
    P[:3, :3] = np.stack([right, down, forward], axis=1)
    P[:3, 3] = RADIUS * d
    return " ".join(repr(float(x)) for x in P.flatten())

@contextmanager
def fake_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        for o, name in enumerate(OBJS):
            obj = Path(tmp) / name
            (obj / "pose").mkdir(parents=True)
            (obj / "rgb").mkdir()
            for v, view in enumerate(VIEWS):
                (obj / "pose" / f"{view}.txt").write_text(pose_text(DIRS[name][v]))
                Image.new("RGBA", (128, 128), color(o, v) + (255,)).save(obj / "rgb" / f"{view}.png")
        if KIND == "raw":
            yield DLDataset(tmp)
        else:
            with tempfile.TemporaryDirectory() as out:
                pack(tmp, out, "train", workers=1)
                yield PackedDataset(out)

def draws(ds, n=300, seed=0):
    # n items, alternating over the objects: (idx, obj_id, (obj, view) of src, (obj, view) of tgt, d_pose)
    random.seed(seed)
    for i in range(n):
        idx = i % len(ds)
        img1, img2, d_pose, obj_id = ds[idx]
        yield idx, obj_id, decode(img1), decode(img2), d_pose.numpy()

def test_fixture_views_come_from_the_same_object():
    with fake_dataset() as ds:
        for idx, obj_id, (o1, _), (o2, _), _ in draws(ds):
            assert o1 == o2 == idx and OBJS[idx] == obj_id, f"item {idx} ({obj_id}) mixed objects {o1} and {o2}"

def test_fixture_two_views_are_different_and_order_is_random():
    with fake_dataset() as ds:
        seen = set()
        for idx, _, (_, v1), (_, v2), _ in draws(ds, n=600):
            assert v1 != v2
            seen.add((idx, v1, v2))
        assert len(seen) == len(OBJS) * 5 * 4, f"only {len(seen)} of the 40 ordered (object, src, tgt) pairs appeared"

def test_fixture_source_image_goes_with_first_half_of_d_pose():
    # most important one: a swapped order would train the model on the wrong pairing
    with fake_dataset() as ds:
        for idx, obj_id, (_, v1), (_, v2), d_pose in draws(ds):
            assert np.allclose(d_pose[:3], DIRS[obj_id][v1], atol=1e-6), f"{obj_id}: source image is view {v1}, first half of d_pose is not its direction"
            assert np.allclose(d_pose[3:], DIRS[obj_id][v2], atol=1e-6), f"{obj_id}: target image is view {v2}, second half of d_pose is not its direction"

def test_fixture_each_half_of_d_pose_has_unit_length():
    with fake_dataset() as ds:
        for *_, d_pose in draws(ds):
            assert abs(np.linalg.norm(d_pose[:3]) - 1) < 1e-6 and abs(np.linalg.norm(d_pose[3:]) - 1) < 1e-6

def test_fixture_seeded_draw_is_reproducible():
    with fake_dataset() as ds:
        random.seed(123)
        a = ds[0]
        random.seed(123)
        b = ds[0]
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and a[3] == b[3]

def test_fixture_pole_cameras():
    # obj_a views 0 and 3 are exactly at +z and -z: finite, exactly the pole direction, and opposite to each other
    with fake_dataset() as ds:
        ds.views[0] = [VIEWS[0], VIEWS[3]]   # restrict the draw to the two poles (the order is still random)
        random.seed(0)
        for _ in range(20):
            img1, img2, d_pose, _ = ds[0]
            (_, v1), (_, v2) = decode(img1), decode(img2)
            assert {v1, v2} == {0, 3}
            assert torch.isfinite(d_pose).all()
            expected = np.array(DIRS["obj_a"][v1] + DIRS["obj_a"][v2], dtype=float)
            assert np.allclose(d_pose.numpy(), expected, atol=1e-6)
            assert np.linalg.norm(d_pose[:3].numpy() - d_pose[3:].numpy()) > 1.99

# ---------- real data: small random sample of objects ----------

@lru_cache(maxsize=None)
def real_ds_of(kind):
    return DLDataset(ROOT) if kind == "raw" else PackedDataset(PACKED)

def real_ds():
    return real_ds_of(KIND)

def sample_indices(n, seed=0):
    return sorted(random.Random(seed).sample(range(len(real_ds())), n))

def test_real_contract():
    ds = real_ds()
    random.seed(0)
    for idx in sample_indices(20):
        img1, img2, d_pose, obj_id = ds[idx]
        assert img1.shape == img2.shape == (3, 64, 64) and d_pose.shape == (6,)
        assert img1.dtype == img2.dtype == d_pose.dtype == torch.float32
        assert isinstance(obj_id, str) and obj_id == ds.objects[idx].name

def test_real_values_in_range_and_reach_white():
    ds = real_ds()
    random.seed(0)
    for idx in sample_indices(20):
        for img in ds[idx][:2]:
            assert img.min() >= -1 and img.max() <= 1, f"{ds.objects[idx].name}: range [{img.min()}, {img.max()}]"
            assert img.max() > 0.99, f"{ds.objects[idx].name}: no near-white pixel, max {img.max()}"

def test_real_dataloader_with_workers():
    ds = real_ds()
    loader = DataLoader(Subset(ds, sample_indices(8)), batch_size=4, num_workers=2)
    batches = list(loader)
    assert len(batches) == 2
    for img1, img2, d_pose, obj_id in batches:
        assert img1.shape == img2.shape == (4, 3, 64, 64) and d_pose.shape == (4, 6) and len(obj_id) == 4

def read_image(png):
    # independent of load_rgb: composite over white, 2x2 box average, scale to [-1, 1]; returns [3, 64, 64]
    a = np.asarray(Image.open(png).convert("RGBA")).astype(np.float64)
    alpha = a[..., 3:] / 255
    rgb = a[..., :3] * alpha + 255 * (1 - alpha)
    rgb = rgb.reshape(64, 2, 64, 2, 3).mean(axis=(1, 3))
    return (rgb / 127.5 - 1).transpose(2, 0, 1)

def read_direction(txt):
    # independent of txt_to_matrix / get_cam_unit_vector: the file is a row-major 4x4, translation is entries 3, 7, 11
    nums = [float(s) for s in Path(txt).read_text().split()]
    t = np.array([nums[3], nums[7], nums[11]])
    return t / np.sqrt((t ** 2).sum())

def test_real_loader_matches_independent_reread():
    # find which file each returned image came from by re-reading every view of the object, then compare poses
    ds = real_ds()
    random.seed(0)
    for idx in sample_indices(5):
        obj = ROOT / ds.objects[idx].name
        names = sorted(txt.stem for txt in (obj / "pose").glob("*.txt") if (obj / "rgb" / f"{txt.stem}.png").exists())
        images = np.stack([read_image(obj / "rgb" / f"{n}.png") for n in names])
        dirs = np.stack([read_direction(obj / "pose" / f"{n}.txt") for n in names])
        img1, img2, d_pose, obj_id = ds[idx]
        assert obj_id == obj.name
        src, tgt = (int(np.abs(images - img.numpy()).max(axis=(1, 2, 3)).argmin()) for img in (img1, img2))
        for img, k in ((img1, src), (img2, tgt)):
            assert np.abs(images[k] - img.numpy()).max() < IMG_TOL[KIND], f"{obj.name}: returned image matches no view"
        assert src != tgt, f"{obj.name}: both images are view {names[src]}"
        assert np.allclose(d_pose[:3].numpy(), dirs[src], atol=1e-6), f"{obj.name}: source image is view {names[src]} but d_pose[:3] is another direction"
        assert np.allclose(d_pose[3:].numpy(), dirs[tgt], atol=1e-6), f"{obj.name}: target image is view {names[tgt]} but d_pose[3:] is another direction"

if __name__ == "__main__":   # the guard also keeps DataLoader workers (spawn) from re-running the tests
    tests = [fn for name, fn in dict(globals()).items() if name.startswith("test_")]
    failed = 0
    for KIND in ("raw", "packed"):
        for fn in tests:
            try:
                fn()
                print(f"PASS [{KIND}]", fn.__name__)
            except AssertionError as e:
                failed += 1
                print(f"FAIL [{KIND}]", fn.__name__)
                print(e)
    sys.exit(1 if failed else 0)
