# Testing and tools

Every test and tool in the repository, what it proves, and the command. Run commands from the project root; prefix
`python` with `.venv/bin/` if the environment is not activated.

pytest is not installed. Each file in `tests/` has its own runner that prints `PASS` / `FAIL` per test and exits non-zero
if any test fails. The test functions are plain `test_*` functions with asserts, so pytest would collect them too (except
`equivalence.py`, whose name has no `test_` prefix).

## Which one do I run?

| Situation | Run |
|---|---|
| I just packed a split and want to know it is right | `tests/equivalence.py`, then `tests/test_pack.py` |
| I changed the loader (`SNRDataset.py`) | `tests/test_loader.py`, `tests/equivalence.py` |
| I changed the packer (`pack_dataset.py`) | `tests/test_pack.py`, then pack a few cars with `--limit` and run `tests/equivalence.py` on the real pack |
| I got new or re-downloaded raw data | `tests/test_data.py` |
| Everything, before a release or a long training run | the loop below, about 1.5 minutes |

```bash
for t in test_data test_loader test_pack equivalence; do python tests/$t.py || break; done
```

The tests sample cars, not the whole dataset, so they stay fast. The one deliberate full scan is `tests/test_data.py`
(all 537,750 poses, about 55 s). Files they need: `cars_train/` for `test_data` and the real-data tests, `packed/` (a
finished `pack_dataset.py --split train`) for `equivalence` and the packed half of `test_loader`.

## Tests (pass / fail)

### `tests/test_data.py`: the raw data is what we assume (4 tests, about 55 s)

```bash
python tests/test_data.py
```

| Test | Proves |
|---|---|
| `test_pose_and_rgb_names_match` | for every car (minus the exclusion list) the `pose/*.txt` names equal the `rgb/*.png` names. This is the check that found the broken car |
| `test_radius_is_same_for_all_views` | the camera radius is the same for all 537,750 views (spread under 1e-5; measured 6e-7) |
| `test_rotation_block_is_proper_rotation` | `R^T R = I` and `det(R) = +1` for every view |
| `test_camera_forward_axis_points_to_origin` | `R[:, 2]` points toward the origin for every view |

### `tests/test_loader.py`: the loader keeps its promises (10 tests, each run on both loaders, about 30 s)

```bash
python tests/test_loader.py
```

The runner runs the 10 tests twice: once on `DLDataset` and once on `PackedDataset` (20 results). Under pytest only the
default (`DLDataset`) is exercised. The 6 fixture tests need no real data.

**Fixture tests.** A fake dataset of 2 cars x 5 views is written to a temporary folder. Every view is a flat-colour image
that encodes its car and view number, and every view has a pose with a known direction. So a test can ask exactly which
view a returned image came from and which view its direction came from, and require them to agree.

| Test | Proves |
|---|---|
| `test_fixture_views_come_from_the_same_object` | both images and the returned `obj_id` belong to the same car |
| `test_fixture_two_views_are_different_and_order_is_random` | the two views differ, and all 40 ordered (car, source, target) pairs occur |
| `test_fixture_source_image_goes_with_first_half_of_d_pose` | source image goes with `d_pose[:3]` and target image with `d_pose[3:]`. The most important one: a swapped order would train the model on the wrong pairing |
| `test_fixture_each_half_of_d_pose_has_unit_length` | each half has length 1 |
| `test_fixture_seeded_draw_is_reproducible` | the same seed gives the same item |
| `test_fixture_pole_cameras` | cameras exactly at +z and -z give finite, exact, opposite directions |

**Real-data tests** (a seeded sample of cars):

| Test | Proves |
|---|---|
| `test_real_contract` | shapes `[3,64,64]` twice and `[6]`, float32, a string `obj_id` |
| `test_real_values_in_range_and_reach_white` | values in `[-1, 1]` and every image reaches near 1 (white background) |
| `test_real_dataloader_with_workers` | a DataLoader with 2 workers produces `[B,3,64,64]` batches |
| `test_real_loader_matches_independent_reread` | the test re-reads all views of a car with its own png and pose parsing, finds which file each returned image came from, and checks the returned directions belong to those same views |

### `tests/test_pack.py`: the packer and `PackedDataset` (17 tests, about 3 s)

```bash
python tests/test_pack.py
```

Most run on a fake split of random-noise images (noise, unlike flat colours, shows a transposed or channel-swapped image).

| Test | Proves |
|---|---|
| `test_shapes_dtypes_and_manifest` | array shapes and dtypes, manifest contents, exactly three output files |
| `test_images_are_box_average_rounded_half_up` | images equal an independent calculation of the 2 x 2 average rounded half up |
| `test_directions_match_known_directions` | directions equal the known ones and have length 1 |
| `test_output_does_not_depend_on_worker_count` | 1 and 2 workers give byte-identical files |
| `test_exclusion_list_drops_object_and_is_recorded` | an excluded car is absent and the manifest records the list |
| `test_limit_takes_first_sorted_objects` | `--limit` takes the first cars in sorted order |
| `test_fails_when_pose_and_rgb_names_differ` | stops, names the car, leaves no files behind |
| `test_fails_when_split_is_ragged` | stops when cars have different view counts |
| `test_fails_when_view_count_is_not_the_expected_one` | `--views N` is enforced |
| `test_fails_when_alpha_is_not_255` | one pixel with alpha 254 stops the run |
| `test_fails_when_radius_is_wrong` | a camera at 1.313 instead of 1.3 stops the run |
| `test_fails_when_there_are_no_objects` | an empty split stops the run |
| `test_packed_dataset_opens_lazily_and_pickles_small` | the arrays are not opened in `__init__` and do not travel with a pickled dataset |
| `test_packed_dataset_gives_no_warnings_private_contiguous_copies` | no warnings; returned tensors are contiguous and private copies |
| `test_packed_dataset_image_layout_matches_noise_images` | height, width and channels are the right way round |
| `test_packed_dataset_real_pickle_is_small` | the same pickle check on the real packed data |
| `test_real_objects_match_loader_and_independent_reread` | 3 real cars packed and compared with an independent re-read of the original files |

### `tests/equivalence.py`: packed data and the original loader describe the same thing (4 tests, about 5 s)

```bash
python tests/equivalence.py
```

Needs the finished `packed/train_*` and `cars_train/`. For the first car, the last car and 18 seeded random ones:

| Test | Proves |
|---|---|
| `test_object_order_is_sorted_folder_order` | the manifest's car order is the sorted folder order and the loader's order |
| `test_view_k_is_view_k_in_the_file_names` | slot `k` is `view_names[k]`, and the file names match the manifest |
| `test_images_and_directions_match_the_loader` | for a pair made of the first and last view and a pair of two random views, both images (within one 8-bit step, 0.0078 in `[-1, 1]`) and both directions (within 1e-6) match the packed slots of exactly those views |
| `test_packed_dataset_returns_same_items_as_raw_loader` | with the same seed, `DLDataset` and `PackedDataset` return the same items |

It prints the largest differences it saw: image 0.00392 (half a grey level), direction 0.

## Tools (they print or save something; nothing to pass or fail)

| Command | What it does |
|---|---|
| `python tools/smoke_batch.py --packed packed` | loads one batch through a DataLoader and prints shapes and min / max. `--root cars_train` does the same for the original files |
| `python tools/check_alpha.py [--n 300] [--save 6] [--out outputs/alpha_check]` | on a random sample: modes and sizes, min / max alpha, whether alpha strictly between 0 and 255 exists, the (0, 0) corner pixel, raw RGB range, range of `load_rgb`; saves a few composites (left: on white, right: alpha) for looking at the edges. Result: alpha 255 everywhere, soft one-pixel edges blending into white, no halo |
| `python tools/check_grid.py [--rows 8] [--out outputs/loader_grid.png]` | saves a picture of loader pairs: source, target, and both directions under them, plus two forced pole pairs. The target should show the same car from another side. A reviewed copy is `docs/img/loader_grid.png` |
| `python tools/time_save_test.py --subset 256 --batch-size 32 --workers 0 2 4 8 --epochs 3 --step-time 0.25` | times the original loader against `PackedDataset` on a subset of cars, same cars for both: per-item time, then samples per second and milliseconds per batch for each worker count, time saved per 1000 batches, and (with `--step-time`, seconds per training step) whether the loader delivers a batch faster than the step consumes it |
| `python pack_dataset.py ...` | builds the packed arrays. Not a test, see [DATA.md](DATA.md#packing-commands) |

For `time_save_test.py`: "first pass" is the first epoch (includes worker start-up and cold files), "steady" is the median
of the later epochs (files in the OS cache). Run it on an otherwise idle machine. Starting workers takes seconds each
(spawn re-imports torch), so the sweep over several worker counts takes a couple of minutes; use `--workers 4 --epochs 2`
for a quick one.

## How the tests themselves were checked

Each group was run against deliberately broken copies of the code, to confirm it fails on them: swapped source and
target, a target taken from the wrong view or the wrong car, wrong scaling (255 instead of 127.5), transposed images,
channel-reversed images, arrays opened in `__init__`, memory maps pickled with the dataset. All were caught. One variant
cannot be: removing the explicit copy out of the memory map changes nothing observable (the transpose that follows
already makes a private copy), so that copy is kept for clarity. These scratch scripts are not kept in the repository.
