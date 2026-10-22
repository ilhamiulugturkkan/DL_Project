"""Tests for pack_dataset.py. Most run on a tiny fake split of noise images; the last one packs a few real objects.
Run from anywhere: python tests/test_pack.py"""
import json
import pickle
import random
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack_dataset
from pack_dataset import pack
from SRNDataset import PackedDataset
from test_loader import OBJS, VIEWS, DIRS, RADIUS, pose_text, read_image, read_direction

ROOT = Path(__file__).resolve().parent.parent / "cars_train"
PACKED = Path(__file__).resolve().parent.parent / "packed"

@contextmanager
def fake_split(seed=0):
    # <tmp>/cars_train with 2 objects x 5 views of random-noise RGBA images (alpha 255); yields (data_dir, noise)
    # where noise[(object, view)] is the [128, 128, 3] uint8 content of that image
    rng = np.random.default_rng(seed)
    noise = {}
    with tempfile.TemporaryDirectory() as tmp:
        for o, name in enumerate(OBJS):
            obj = Path(tmp) / "cars_train" / name
            (obj / "pose").mkdir(parents=True)
            (obj / "rgb").mkdir()
            for v, view in enumerate(VIEWS):
                (obj / "pose" / f"{view}.txt").write_text(pose_text(DIRS[name][v]))
                arr = rng.integers(0, 256, (128, 128, 4), dtype=np.uint8)
                arr[..., 3] = 255
                Image.fromarray(arr, "RGBA").save(obj / "rgb" / f"{view}.png")
                noise[(name, view)] = arr[..., :3]
        yield Path(tmp), noise

def run(data_dir, out, **kwargs):
    return pack(data_dir / "cars_train", out, "train", **kwargs)

def expect_failure(data_dir, *needles, **kwargs):
    # packing must fail with a message containing every needle, and leave nothing in the output folder
    with tempfile.TemporaryDirectory() as out:
        try:
            run(data_dir, Path(out), **kwargs)
        except (ValueError, RuntimeError) as e:
            for needle in needles:
                assert needle in str(e), f"{needle!r} missing from error: {e}"
            assert list(Path(out).iterdir()) == [], f"left files behind: {list(Path(out).iterdir())}"
            return
        raise AssertionError("packing should have failed")

def test_shapes_dtypes_and_manifest():
    with fake_split() as (data, _), tempfile.TemporaryDirectory() as out:
        files = run(data, Path(out))
        images = np.load(files["images"], mmap_mode="r")
        directions = np.load(files["directions"])
        assert images.shape == (2, 5, 64, 64, 3) and images.dtype == np.uint8
        assert directions.shape == (2, 5, 3) and directions.dtype == np.float32
        manifest = json.loads(files["manifest"].read_text())
        assert manifest["object_ids"] == OBJS and manifest["view_names"] == VIEWS
        assert manifest["n_objects"] == 2 and manifest["n_views_per_object"] == 5 and manifest["n_images"] == 10
        assert manifest["images"]["shape"] == [2, 5, 64, 64, 3] and manifest["directions"]["shape"] == [2, 5, 3]
        assert manifest["background"] == [255, 255, 255] and manifest["resize"] and manifest["date"]
        assert sorted(p.name for p in Path(out).iterdir()) == ["train_directions.npy", "train_images.npy", "train_manifest.json"]

def test_images_are_box_average_rounded_half_up():
    with fake_split() as (data, noise), tempfile.TemporaryDirectory() as out:
        images = np.load(run(data, Path(out))["images"])
        for o, name in enumerate(OBJS):
            for v, view in enumerate(VIEWS):
                mean = noise[(name, view)].astype(np.float64).reshape(64, 2, 64, 2, 3).mean(axis=(1, 3))
                assert np.array_equal(images[o, v], np.floor(mean + 0.5).astype(np.uint8)), f"{name}/{view}"

def test_directions_match_known_directions():
    with fake_split() as (data, _), tempfile.TemporaryDirectory() as out:
        directions = np.load(run(data, Path(out))["directions"])
        for o, name in enumerate(OBJS):
            assert np.allclose(directions[o], DIRS[name], atol=1e-6)
            assert np.allclose(np.linalg.norm(directions[o], axis=1), 1, atol=1e-6)

def test_output_does_not_depend_on_worker_count():
    with fake_split() as (data, _), tempfile.TemporaryDirectory() as out1, tempfile.TemporaryDirectory() as out2:
        a = run(data, Path(out1), workers=1)
        b = run(data, Path(out2), workers=2)
        for key in ("images", "directions"):
            assert a[key].read_bytes() == b[key].read_bytes(), key

def test_exclusion_list_drops_object_and_is_recorded():
    saved = pack_dataset.EXCLUDED_OBJECTS
    pack_dataset.EXCLUDED_OBJECTS = {"obj_b"}
    try:
        with fake_split() as (data, _), tempfile.TemporaryDirectory() as out:
            files = run(data, Path(out))
            manifest = json.loads(files["manifest"].read_text())
            assert manifest["object_ids"] == ["obj_a"] and np.load(files["images"]).shape[0] == 1
            assert manifest["exclusion_list"] == ["obj_b"] and manifest["excluded_present_in_split"] == ["obj_b"]
    finally:
        pack_dataset.EXCLUDED_OBJECTS = saved

def test_limit_takes_first_sorted_objects():
    with fake_split() as (data, _), tempfile.TemporaryDirectory() as out:
        files = run(data, Path(out), limit=1)
        assert json.loads(files["manifest"].read_text())["object_ids"] == ["obj_a"]

def test_fails_when_pose_and_rgb_names_differ():
    with fake_split() as (data, _):
        (data / "cars_train" / "obj_b" / "rgb" / "000002.png").unlink()
        expect_failure(data, "obj_b", "000002")

def test_fails_when_split_is_ragged():
    with fake_split() as (data, _):
        for sub, ext in (("pose", "txt"), ("rgb", "png")):
            (data / "cars_train" / "obj_b" / sub / f"000004.{ext}").unlink()   # names still match, count differs
        expect_failure(data, "obj_b", "ragged")

def test_fails_when_view_count_is_not_the_expected_one():
    with fake_split() as (data, _):
        expect_failure(data, "expected 250", expect_views=250)

def test_fails_when_alpha_is_not_255():
    with fake_split() as (data, _):
        png = data / "cars_train" / "obj_a" / "rgb" / "000001.png"
        arr = np.array(Image.open(png))
        arr[5, 5, 3] = 254
        Image.fromarray(arr, "RGBA").save(png)
        expect_failure(data, "obj_a", "000001", "alpha")

def test_fails_when_radius_is_wrong():
    with fake_split() as (data, _):
        txt = data / "cars_train" / "obj_b" / "pose" / "000003.txt"
        nums = [float(s) for s in txt.read_text().split()]
        for i in (3, 7, 11):
            nums[i] *= 1.01
        txt.write_text(" ".join(repr(x) for x in nums))
        expect_failure(data, "obj_b", "000003", "radius")

def test_fails_when_there_are_no_objects():
    with tempfile.TemporaryDirectory() as data:
        (Path(data) / "cars_train").mkdir()
        expect_failure(Path(data), "no objects")

def test_packed_dataset_opens_lazily_and_pickles_small():
    # DataLoader workers get the dataset pickled: the memmaps must not be opened yet, and must not travel with it
    with fake_split() as (data, _), tempfile.TemporaryDirectory() as out:
        run(data, Path(out))
        ds = PackedDataset(out)
        assert ds._images is None and ds._directions is None, "arrays opened in __init__"
        ds[0]
        assert ds._images is not None
        blob = pickle.dumps(ds)
        clone = pickle.loads(blob)
        assert clone._images is None and clone._directions is None
        assert len(blob) < 100_000, f"pickled dataset is {len(blob)} bytes"
        assert clone[0][3] == ds[0][3]   # and the clone opens its own maps

def test_packed_dataset_gives_no_warnings_private_contiguous_copies():
    with fake_split() as (data, _), tempfile.TemporaryDirectory() as out:
        run(data, Path(out))
        ds = PackedDataset(out)
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # e.g. torch's "NumPy array is not writable" for a read-only memmap
            for _ in range(20):
                ds[0]
        random.seed(0)
        img1, img2, d_pose, _ = ds[0]
        assert img1.is_contiguous() and img2.is_contiguous() and d_pose.is_contiguous()
        img1.add_(5), img2.add_(5), d_pose.add_(5)   # scribble on the returned tensors
        random.seed(0)
        again = ds[0]   # same draw, must be untouched
        assert again[0].max() <= 1 and again[1].max() <= 1 and abs(again[2][:3].norm().item() - 1) < 1e-6

def test_packed_dataset_image_layout_matches_noise_images():
    # flat-color fixtures can't show a transposed or channel-swapped image, noise can
    with fake_split() as (data, noise), tempfile.TemporaryDirectory() as out:
        run(data, Path(out))
        ds = PackedDataset(out)
        ds.views[0] = [VIEWS[1], VIEWS[3]]   # the draw is these two, in random order
        random.seed(0)
        img1, img2, _, _ = ds[0]

        def expected(view):   # [3, 64, 64] in [-1, 1], computed here from the raw noise
            mean = noise[("obj_a", view)].astype(np.float64).reshape(64, 2, 64, 2, 3).mean(axis=(1, 3))
            return (np.floor(mean + 0.5) / 127.5 - 1).transpose(2, 0, 1)

        e1, e3 = expected(VIEWS[1]), expected(VIEWS[3])
        assert not np.allclose(e1, e3)
        same = lambda got, want: np.allclose(got.numpy(), want, atol=1e-6)
        assert (same(img1, e1) and same(img2, e3)) or (same(img1, e3) and same(img2, e1)), "returned images are not the packed noise images"

def test_packed_dataset_real_pickle_is_small():
    ds = PackedDataset(PACKED)
    ds[0]
    assert len(pickle.dumps(ds)) < 5_000_000, "pickled dataset carries the arrays"

def test_real_objects_match_loader_and_independent_reread():
    with tempfile.TemporaryDirectory() as out:
        files = pack(ROOT, Path(out), "train", workers=2, limit=3)
        images = np.load(files["images"], mmap_mode="r")
        directions = np.load(files["directions"])
        manifest = json.loads(files["manifest"].read_text())
        assert images.shape == (3, 250, 64, 64, 3) and manifest["n_views_per_object"] == 250
        for o, name in enumerate(manifest["object_ids"]):
            obj = ROOT / name
            for k in (0, 17, 249):
                view = manifest["view_names"][k]
                # packed uint8 -> [-1, 1] differs from the float pipeline only by the 8-bit rounding (at most half a grey level)
                packed = images[o, k].astype(np.float64) / 127.5 - 1
                reference = read_image(obj / "rgb" / f"{view}.png").transpose(1, 2, 0)
                assert np.abs(packed - reference).max() <= 0.5 / 127.5 + 1e-9, f"{name}/{view}"
                assert np.allclose(directions[o, k], read_direction(obj / "pose" / f"{view}.txt"), atol=1e-6)

if __name__ == "__main__":   # the guard also keeps the spawned pack workers from re-running the tests
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
