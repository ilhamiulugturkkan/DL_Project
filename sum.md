# Data Pipeline Summary (SRN Cars)

## Goal
Each training example is a pair of views of the **same object**:
`(src_img, tgt_img, d_pose, obj_id)`

## What we learned about the data
- 250 views per object in `cars_train` (`rgb/`, `pose/`, `intrinsics/` per view).
- Images are **128x128 RGBA** (matches the 3DiM description).
- Pose files are 4x4 **camera-to-world** matrices; camera position = `pose[:3, 3]`.
- Camera distance from the origin is constant (radius 1.3), so direction alone is enough.
- Cameras cover the **whole sphere, including the poles**, so azimuth/elevation is unstable there.
- Intrinsics are ignored by the model.

## Decisions
| Topic | Choice |
|---|---|
| Pose conditioning | **Option B**: source and target camera unit vectors (6 numbers) instead of (dtheta, dphi) |
| Background | RGBA composited onto white, **before** resizing |
| Downsampling | 128 -> 64 with **box/area averaging** (`avg_pool2d`, kernel 2) |
| Value range | **[-1, 1]**, as in DDPM-style work |
| Pairs per object | 250 x 249 = 62,250 ordered pairs (sampled at random, never listed) |
| Dataset index | One index = one object; two distinct random views picked inside `__getitem__` |

## Current code (`SNRDataset.py`)
- `DLDataset` returns `(src_img [3,64,64], tgt_img [3,64,64], d_pose [6], obj_id)`.
- Smoke test passed: shapes `[8,3,64,64]`, `[8,3,64,64]`, `[8,6]`, 8 object IDs.