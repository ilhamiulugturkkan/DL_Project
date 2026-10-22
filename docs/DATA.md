# Data reference

Formats and exact processing. For why things were chosen see [DECISIONS.md](DECISIONS.md); for how it is checked see
[TESTING.md](TESTING.md).

## Raw split (input)

```
cars_train/                       one folder per car (cars_val/, cars_test/ the same)
  <object_id>/
    rgb/<view>.png                128 x 128 RGBA, e.g. 000000.png ... 000249.png
    pose/<view>.txt               one line, 16 numbers: a 4 x 4 camera-to-world matrix, row-major
    intrinsics/<view>.txt         not used
    intrinsics.txt                not used
```

A pose file reads as

```
| r11 r12 r13 | c1 |
| r21 r22 r23 | c2 |
| r31 r32 r33 | c3 |
|  0   0   0  | 1  |
```

where `(c1, c2, c3) = pose[:3, 3]` is the camera position. The camera is 1.3 from the origin, looks at the origin
(`R[:, 2] = -position / |position|`) and the world z axis is up.

Folders whose name starts with `.` are ignored. The cars listed in `EXCLUDED_OBJECTS` (`SNRDataset.py`) are skipped.

## Packed split (output of `pack_dataset.py`)

```
packed/
  train_images.npy        [cars, views, 64, 64, 3]   uint8     0-255, RGB, channels last
  train_directions.npy    [cars, views, 3]           float32   unit vector from the origin to the camera
  train_manifest.json     what is in the two arrays, and how they were made
```

(`val_*` and `test_*` the same way.) Slot `[o, v]` is car `object_ids[o]`, view `view_names[v]`, in both arrays. Cars are in
sorted folder-name order and views in sorted file-name order. For the train split: 2,150 cars x 250 views, 6.6 GB of images.

Read them directly:

```python
import json, numpy as np
images = np.load("packed/train_images.npy", mmap_mode="r")      # no 6.6 GB read into RAM
directions = np.load("packed/train_directions.npy")
manifest = json.load(open("packed/train_manifest.json"))
x = images[o, v].astype(np.float32) / 127.5 - 1                 # [64, 64, 3] in [-1, 1]
```

or, as the training code should, through `PackedDataset` (see below).

### Manifest

`train_manifest.json` is plain JSON, scalars first:

| Field | Content |
|---|---|
| `split`, `date`, `source`, `limit` | which split, when it was built, the folder it was built from, and `--limit` if used |
| `n_objects`, `n_views_per_object`, `n_images` | counts |
| `images`, `directions` | `{file, shape, dtype}` of each array |
| `resize`, `background`, `radius`, `radius_tolerance` | how it was made (box average, white, 1.3, 1e-5) |
| `exclusion_list`, `excluded_present_in_split` | the exclusion list that was applied, and which of its cars exist in this split |
| `object_ids` | the car names in slot order |
| `view_names` | the view names in slot order, shared by all cars. If cars ever differ it is `view_names_per_object` instead |

## What is done to each view

1. **Composite** the RGBA png onto white (255, 255, 255).
2. **Reduce 128 to 64**: average each 2 x 2 block of the 0-255 values, as an integer sum, and round half up to `uint8` (`(sum + 2) >> 2`).
3. **Direction** = `pose[:3, 3] / |pose[:3, 3]|`, stored as `float32`.

At load time the image becomes `[3, 64, 64]` float32 with `x / 127.5 - 1`.

**The only lossy step is the rounding to 8 bits in step 2.** A 2 x 2 average is a multiple of 0.25, so rounding moves a
value by at most half a grey level (0.0039 in `[-1, 1]`). Packed images therefore differ from what `DLDataset` computes in
float by at most that (measured: 0.00392). Compositing changes nothing on this data because alpha is 255 everywhere.

## Checks done while packing

Any failure stops the run and names the car (and view). The only way a car is left out is `EXCLUDED_OBJECTS`.

- `pose/` and `rgb/` names are identical for every car.
- Every car has the same number of views. The number is detected from the first car, not hard-coded. `--views N` also requires it to equal N.
- Every png is RGBA, 128 x 128, with alpha 255 everywhere.
- Camera radius is 1.3 within 1e-5, and every stored direction has length 1 within 1e-6.
- Every slot of the output was written.

## Datasets

Both live in `SNRDataset.py` and return the same item: `(src_img [3, 64, 64], tgt_img [3, 64, 64], d_pose [6], obj_id)`,
float32 images in `[-1, 1]`, `d_pose` = source direction then target direction, `obj_id` the car's name.

| | `PackedDataset(packed_dir, split="train")` | `DLDataset(path)` |
|---|---|---|
| reads | the packed arrays | the original png and txt files |
| speed | about 3x faster per batch | reference |
| use for | training | checking, and as the reference the packed data is tested against |
| car list | `manifest["object_ids"]` | sorted folders minus `EXCLUDED_OBJECTS` |
| views of a car | `manifest["view_names"]` | names present in both `pose/` and `rgb/` |

Both keep a `views` list per car and draw a pair with `random.sample(views, 2)`, so the same `random.seed` gives the same
pair in both (up to the 8-bit rounding of the images).

`PackedDataset` opens the arrays on first use, so it can be handed to DataLoader workers cheaply (see
[DECISIONS.md, section 8](DECISIONS.md#8-how-the-packed-data-is-loaded-packeddataset)).

## Packing commands

```bash
python pack_dataset.py --split train                          # cars_train/ -> packed/train_*
python pack_dataset.py --split val --out /scratch/packed --workers 16
python pack_dataset.py --split train --limit 8 --views 250 --out /tmp/packed_tiny
```

| Argument | Default | Meaning |
|---|---|---|
| `--split` | required | `train`, `val` or `test`; reads `<data-dir>/cars_<split>/` |
| `--data-dir` | project folder | where `cars_<split>/` is |
| `--out` | `packed/` | where the three files go |
| `--workers` | number of CPUs | worker processes; `1` runs in the main process |
| `--limit N` | all | only the first N cars (sorted), for quick runs; recorded in the manifest |
| `--views N` | detect | additionally fail unless every car has exactly N views |

The output does not depend on `--workers`. Files are written under `*.tmp` names and renamed at the end, manifest last: a
manifest next to the arrays means the run finished, and a crash leaves no half-filled file that looks complete.
