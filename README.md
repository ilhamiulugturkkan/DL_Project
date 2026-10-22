# SRN Cars data pipeline

Data loading for a novel-view-synthesis diffusion model on the **SRN Cars** dataset (ShapeNet cars, 250 rendered views per
car). The model gets one view of a car plus the two camera positions and has to produce the other view, so every training
example is a **pair of views of the same car**.

This repository turns the raw dataset (about 537,000 small png / txt files) into two big arrays, loads pairs from them
fast, and proves with tests that the images and camera directions in those pairs are the right ones.

## What the loader returns

One index is one car. Each item is two distinct random views of it:

| Item | Shape / type | Meaning |
|---|---|---|
| `src_img` | `[3, 64, 64]` float32, range `[-1, 1]` | view 1, the source |
| `tgt_img` | `[3, 64, 64]` float32, range `[-1, 1]` | view 2, the target |
| `d_pose` | `[6]` float32 | source camera unit direction `(x, y, z)`, then target camera unit direction `(x, y, z)` |
| `obj_id` | `str` | the car's folder name |

The dataset has 2,150 cars with 250 views each (a 2,151st car is excluded, see [Known issues](#known-issues)), so there are
250 x 249 = 62,250 ordered pairs per car. They are drawn at random in `__getitem__` and never listed.

## Quick start

```bash
# 1. environment (Python 3.11)
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. put the raw data next to this file as cars_train/ (cars_val/, cars_test/ if you have them), then build the arrays once
.venv/bin/python pack_dataset.py --split train          # -> packed/train_{images,directions}.npy + train_manifest.json

# 3. check that it worked (about 1 minute)
.venv/bin/python tests/test_pack.py
.venv/bin/python tests/equivalence.py
.venv/bin/python tools/smoke_batch.py --packed packed
```

Use it in training:

```python
from torch.utils.data import DataLoader
from SNRDataset import PackedDataset

ds = PackedDataset("packed", split="train")
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
for src_img, tgt_img, d_pose, obj_id in loader:
    ...
```

Run Python from the project root (or put it on `PYTHONPATH`) so that `import SNRDataset` works.

## What to run to do what

| I want to ... | Run | Time |
|---|---|---|
| build the packed arrays for a split | `python pack_dataset.py --split train` (or `val`, `test`) | minutes, needs about 7 GB free for train |
| try the packer on a few cars first | `python pack_dataset.py --split train --limit 8 --views 250 --out /tmp/packed_tiny` | seconds |
| see that a batch loads, and its shape and range | `python tools/smoke_batch.py --packed packed` | seconds |
| check that the packed arrays agree with the original files | `python tests/equivalence.py` | 5 s |
| test the loader (both versions) | `python tests/test_loader.py` | 30 s |
| test the packer | `python tests/test_pack.py` | 3 s |
| check the raw data is what we assume (radius, rotations, file names) | `python tests/test_data.py` | 55 s, scans all 537,750 poses |
| measure how much time the packed arrays save | `python tools/time_save_test.py --subset 256 --batch-size 32 --workers 4 --step-time 0.25` | about 1 min |
| look at what the loader returns | `python tools/check_grid.py` then open `outputs/loader_grid.png` | seconds |
| look at the alpha channel and the edges | `python tools/check_alpha.py` then open `outputs/alpha_check/` | seconds |

All commands are written from the project root; prefix `python` with `.venv/bin/` if the environment is not activated.
Every test file prints `PASS` / `FAIL` per test and exits non-zero on a failure. Details: [docs/TESTING.md](docs/TESTING.md).

## What is in this repository

```
SNRDataset.py        the datasets: PackedDataset (fast, use for training) and DLDataset (original files, the reference)
pack_dataset.py      builds the packed arrays from a raw split, with strict checks
requirements.txt     Python packages
docs/
  DECISIONS.md       every design decision in order, with the reason, plus what is still open
  DATA.md            formats: raw folders, packed arrays, manifest, how each view is processed, checks
  TESTING.md         every test and tool, what it proves, how to run it
  img/loader_grid.png   a reviewed picture of loader pairs
tests/               pass / fail tests
  test_data.py         the raw data is what we assume
  test_loader.py       loader contract and pair correctness, on a fake dataset and on real data, for both loaders
  test_pack.py         the packer and PackedDataset
  equivalence.py       packed arrays and original loader describe the same thing
tools/               scripts that print or save something
  smoke_batch.py       one batch: shapes and value range
  check_alpha.py       alpha / RGB value ranges and edge pictures
  check_grid.py        picture of loader pairs
  time_save_test.py    old loader against new loader, time per batch
cars_train/          raw data (not in git)
packed/              built arrays (not in git)
outputs/             generated pictures (not in git)
```

`SNRDataset.py` and `pack_dataset.py` stay in the project root because they are the two things other code imports or runs.

## Key decisions

The full list with reasons is in [docs/DECISIONS.md](docs/DECISIONS.md). In short:

1. **Pose conditioning is the two camera unit vectors (6 numbers)**, not angle differences. Cameras cover the whole sphere including the poles, where azimuth and elevation are unstable. The camera radius is constant (1.3) and the camera always looks at the origin, so a direction is all the pose there is.
2. **Images are composited on white, then reduced 128 to 64 by 2x2 box averaging, then scaled to `[-1, 1]`.**
3. **One index is one car and the two views are drawn at random** inside `__getitem__`.
4. **A broken car is excluded by name**, and the loaders only use views that exist as both pose and image.
5. **The raw files are packed once into `uint8` images and `float32` directions.** The only lossy step is rounding to 8 bits after averaging (at most half a grey level, 0.0039 in `[-1, 1]`).
6. **The packer fails loudly and writes atomically**: any surprise in the data stops it and names the car; the output appears under its final name only when complete.
7. **The new loader maps the arrays lazily and copies slices out** so DataLoader workers share the file through the OS cache.
8. **Tests do not trust the code they test**: a fake dataset where every image encodes which view it is, an independent re-read of the original files, and a direct comparison of the packed arrays with the original loader.

## Results

- All 35 tests pass (the 10 loader tests run once per loader, so 45 results in total).
- Packed images differ from the original loader's by at most 0.00392 (half a grey level; the allowed limit is 0.00784). Directions are identical.
- Time per batch of 32, 256-car subset, this machine (10 cores, local SSD, files already in the OS cache):

| workers | old loader | new loader | new loader's gain |
|---|---|---|---|
| 0 | 46.2 ms | 15.5 ms | 3.0x |
| 2 | 20.9 ms | 6.3 ms | 3.3x |
| 4 | 10.5 ms | 3.6 ms | 2.9x |
| 8 | 9.7 ms | 2.9 ms | 3.4x |

  One `__getitem__` in a single process: 1.76 ms old, 0.44 ms new. With 4 workers that is 3,055 against 8,833 samples per second.
  Not measured: a cold or shared file system (for example the cluster), where the old loader should be hurt more, and the
  comparison with the training step time, which needs `--step-time`.

## Known issues

- **Car `4cce557de0c31a0e70a43c2d978e502e` is broken**: 250 poses but only 142 rgb files. It is listed in `EXCLUDED_OBJECTS` in `SNRDataset.py`, and the packer and tests skip it by that list. It is the only such car of 2,151.
- **Validation and test pairs are still random.** `__getitem__` draws a new pair on every call, which is right for training only. Evaluation needs fixed pairs.
- **The `alpha` channel is 255 everywhere**, so the images are effectively RGB with a white background already baked in. Compositing on white changes nothing on this data; it is kept (and checked) so the pipeline would still be correct if that ever changed.
- Other open points are listed at the end of [docs/DECISIONS.md](docs/DECISIONS.md).
