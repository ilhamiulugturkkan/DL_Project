# Decisions

What was decided, in the order the data flows, and why. Each entry says what was chosen, the reason, and where it lives.
"Measured" means we checked it on the real data; the commands are in [TESTING.md](TESTING.md).

## 1. What the raw data is

Facts about `cars_train/` that the rest depends on (all measured):

- 2,151 car folders, each with `rgb/`, `pose/`, `intrinsics/` and `intrinsics.txt`; 250 views per car (one car is broken, see decision 6).
- Images are **128 x 128 RGBA** pngs (this matches the 3DiM description).
- A pose file is one line of 16 numbers: a row-major 4 x 4 **camera-to-world** matrix. The camera position is `pose[:3, 3]`.
- The camera is always **1.3 from the origin** (spread over all 537,750 views: 6e-7) and always **looks at the origin**: the rotation block is a proper rotation (`R^T R = I`, `det = +1`) and its third column `R[:, 2]` equals `-position / |position|`. World **z is up** (the camera's y axis always points toward -z).
- Cameras cover the **whole sphere, including the poles**. About 3% of views are within 18 degrees of a pole.
- **Alpha is 255 in every image** (6,000 images from 2,028 cars checked, corner pixel always opaque white). The background is already white.
- Intrinsics are not used.

## 2. What one training example is

**Choice:** `(src_img, tgt_img, d_pose, obj_id)`: two different views of the same car, source first, target second.
`__getitem__(idx)` picks the car by index and draws the two views at random inside the call.

**Why:** the model sees one view plus the camera change and must produce the other view. Listing all pairs would mean
250 x 249 = 62,250 ordered pairs per car (about 134 million in total), so pairs are sampled, never stored. One index per
car keeps every car equally likely per epoch.

**Consequence:** the random draw is right for training only. Validation and test need fixed pairs (open, see the end).
`PackedDataset` and `DLDataset` draw with the same `random.sample(views, 2)`, so the same seed gives the same pair in both.
Code: `DLDataset.__getitem__`, `PackedDataset.__getitem__`.

## 3. How the camera pose is given to the model

**Choice ("option B"):** `d_pose` is 6 numbers: the **unit vector from the origin to the source camera, then the same
for the target camera**. Not (delta azimuth, delta elevation), not the rotation matrices.

**Why:**
- Cameras go over the poles, where azimuth is undefined and elevation wraps. A unit vector has no such singularity: two cameras near a pole have nearby vectors even when their azimuths differ by 180 degrees. The tests put cameras exactly on the poles and require finite, exact, opposite directions.
- The radius is constant (1.3), so the direction is the whole camera position.
- The rotation `R` carries no extra information because the camera always points at the origin (measured, decision 1), so it is not used.

Code: `get_cam_unit_vector`. The order (source then target) is tested, because a swapped order would silently train the model on the wrong pairing.

## 4. How the images are prepared

**Choice, in this order:**
1. composite the RGBA image onto **white** (255, 255, 255) *before* resizing;
2. reduce **128 to 64** by **box (area) averaging** of 2 x 2 blocks;
3. scale to **[-1, 1]** with `x / 127.5 - 1`.

**Why:** compositing first keeps the resize from mixing in the hidden colour of transparent pixels; box averaging is the
plain area average (no ringing, no aliasing from a naive stride); [-1, 1] is the range DDPM-style work uses.
Alpha is 255 everywhere here (decision 1), so step 1 changes nothing on this data. It stays in the code, and the packer
fails if alpha is ever below 255, so the pipeline cannot be quietly wrong if the data changes.

Code: `load_rgb` (original files) and `pack_image` in `pack_dataset.py` (packed arrays).

## 5. Why the files are packed

**Problem:** one epoch opens about 4,300 pngs and 4,300 txt files, and the whole set is 537,750 + 537,750 small files.

**Choice:** pack each split once into
- `<split>_images.npy`: `[cars, views, 64, 64, 3]`, **uint8**;
- `<split>_directions.npy`: `[cars, views, 3]`, **float32**;
- `<split>_manifest.json`: what is in them and how they were made.

**Why uint8:** the images are 8-bit anyway, and uint8 is 4x smaller than float32 (6.6 GB for train instead of 26 GB),
so the whole array fits in the OS cache of a normal machine. The conversion to float happens per item, in `__getitem__`.

**The only lossy step is rounding to 8 bits after averaging.** A 2 x 2 average is a multiple of 0.25, so rounding moves a
value by at most half a grey level (0.0039 in [-1, 1]). It rounds half up, computed in integers as `(sum + 2) >> 2`, so it
is exact and does not depend on floating point. Measured: packed images differ from the original loader's by at most
0.00392. Everything else (compositing, directions) is exact.

## 6. The broken car, and what the loaders do about it

**Found by:** the test that pose names equal rgb names for every car. Car `4cce557de0c31a0e70a43c2d978e502e` has 250
poses but only 142 rgb files, and it is the only one of 2,151. Without a fix, drawing it would raise `FileNotFoundError`
in a DataLoader worker about two times in three.

**Choice, two layers:**
- an **exclusion list** (`EXCLUDED_OBJECTS` in `SRNDataset.py`) that drops the car by name. It is the only way a car can be skipped, so nothing disappears silently, and the manifest records the list used;
- `DLDataset` builds each car's view list from the **names present in both** `pose/` and `rgb/`, and asserts at least 2 views.

The packer is stricter than the loader: for every car it requires the two name lists to be identical and fails otherwise.
`tests/test_data.py` keeps checking this for every car, which is also what would catch a future bad copy or download.

## 7. How the packer works (`pack_dataset.py`)

| Choice | Why |
|---|---|
| Cars in **sorted** order, each with a **fixed slot** | the same input always gives the same bytes, whatever the number of workers or the order they finish in (tested: 1 and 2 workers give identical files) |
| Output file created on disk as a **memory map**, filled car by car | a 6.6 GB result never needs 6.6 GB of RAM |
| One car per task, several **worker processes**; the workers return a small result and the main process writes it into the car's slot | decoding is the slow part and parallelizes cleanly |
| Written under a **`.tmp` name and renamed at the end, manifest last** | a crash cannot leave a half-filled file that looks complete; a manifest next to the arrays means the run finished |
| **Fail loudly**: names match, same view count in every car, every png is RGBA 128 x 128 with alpha 255, radius 1.3 within 1e-5, directions of length 1 within 1e-6; any failure stops the run and names the car | surprises should stop the run, not become silent training noise |
| View count is **detected** and must be the same for the whole split (`--views N` additionally pins it) | nothing hard-codes 250, but a ragged val or test split stops the run |
| The split is an argument (`--split train/val/test`), `--limit N` runs the first N cars | one script for all splits, and quick trial runs |

## 8. How the packed data is loaded (`PackedDataset`)

- **Same item as `DLDataset`**: same shapes, dtypes and order, so the model code does not change.
- **The arrays are opened lazily, on first use, in the worker** (`mmap_mode="r"`), not in `__init__`. Every worker maps the file itself and the operating system shares the pages, so there is no private 6.6 GB copy per worker.
- **The memory maps are dropped when the dataset is pickled** (`__getstate__`); pickling one would copy the whole array.
- **The two slices are copied out of the map before they become tensors** (`packed_to_tensor`), because turning a read-only memory map into a tensor directly warns and can surprise. The returned tensors are private and contiguous.

**What it bought** (256-car subset, batch 32, this machine, files in the OS cache): about 3x more samples per second at
every worker count (for example 3,055 to 8,833 samples/s with 4 workers), and 0.44 ms instead of 1.76 ms for one item.
Full table in the [README](../README.md#results).

## 9. How the tests were designed

The worst failure in this pipeline is an image paired with the wrong camera, or source and target swapped: the shapes are
right, the loss goes down, and the model learns the wrong thing. No shape check can see that, so the tests do not trust
the code they check:

1. **A fake dataset where every image encodes which view it is** (a flat grey level) and every pose has a known direction. A test can then ask exactly which view the returned source image came from and which pose it came from, and require that they agree. It runs in under a second and needs no real data, so it runs anywhere. Cameras placed exactly on the poles are included.
2. **Independent re-reads**: the tests re-read the pngs and pose files with their own code (not the loader's helpers) and compare with what the loader returned.
3. **Equivalence**: the packed arrays are compared, view by view including the first and last view of each car, with the original loader's output, and the object order and view names are checked against the folders.
4. **Every test was also run against deliberately broken code** (swapped order, wrong view or car, wrong scaling, transposed or channel-reversed images, arrays opened too early, ...) to confirm it fails. One broken variant, removing an explicit copy that is redundant anyway, changes nothing observable and cannot be caught.

Tests use **a sample of cars**, not the whole dataset, except `tests/test_data.py`, which is the one deliberate full scan of
the raw poses.

## What is still open

- **Fixed validation and test pairs.** The random draw in `__getitem__` is for training only.
- **`cars_val` and `cars_test` have not been packed here.** `pack_dataset.py --split val` and `--split test` are written and tested on fake data, but they have not been run on those splits.
- **The cost against the training step is unknown.** `tools/time_save_test.py --step-time <seconds>` prints whether the loader delivers a batch faster than the step consumes it. At 4 workers the new loader needs 3.6 ms per batch of 32 and the old one 10.5 ms, so the loader is the bottleneck only if a step takes less than that.
- **Cold or shared file systems are not measured.** The timings are on a local SSD with the files already cached; the old loader should lose more on a cold network file system.
- **Possibly shared camera layout.** In the first 50 cars, 30 had views near the south pole but none near the north pole, and two different cars drew the identical direction in the grid. That suggests the 250 camera positions are shared between cars. It was not verified.

## Appendix: original working notes (kept as written)

`notes.md`, before this repository was reorganized:

```text
cat cars_train/100715345ee54d7ae38b52b4ee9d36a3/pose/000000.txt:
-0.5042614340782166 -0.37587812542915344 0.7774547934532166 -1.0106911659240723
-0.8635510206222534 0.21949009597301483 -0.4539864957332611 0.5901824235916138 
-5.96046447753903e-08 -0.9002998471260071 -0.43527036905288696 0.5658514499664307 
0.0 0.0 -0.0 1.0

0.49751347303390503 -0.5331316590309143 0.6842886805534363 -0.889575183391571
-0.8674562573432922 -0.3057679533958435 0.39246100187301636 -0.5101994872093201
1.6391277313232422e-07 -0.7888451814651489 -0.6145920753479004 0.7989696860313416
-0.0 0.0 -0.0 1.0

| r11 r12 r13 | c1 |
| r21 r22 r23 | c2 |
| r31 r32 r33 | c3 |
|  0   0   0  | 1  |

pose[:3, 3] 

R's are not that important as camera is always pointing to the center.


4cce557de0c31a0e70a43c2d978e502e has 250 poses but only 142 rgb files, and it's the only one of 2,151 objects like that.
```
