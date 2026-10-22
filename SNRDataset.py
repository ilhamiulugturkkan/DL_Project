"""Datasets for novel-view-synthesis training on SRN Cars.

One index is one object. Each item is a pair of two distinct random views of that object:

    (src_img [3, 64, 64] float32 in [-1, 1], tgt_img [3, 64, 64], d_pose [6] float32, obj_id str)

d_pose is the source camera's unit direction (xyz) followed by the target camera's unit direction (xyz).
View 1 is the source, view 2 is the target.

    DLDataset       reads the original png / txt files. Slow, kept as the reference.
    PackedDataset   reads the arrays written by pack_dataset.py. Fast, use this for training.

See docs/DATA.md for the formats and docs/DECISIONS.md for why things are the way they are.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# Objects that are never used. 4cce557d...: only 142 of its 250 views have an rgb png.
EXCLUDED_OBJECTS = {
    "4cce557de0c31a0e70a43c2d978e502e",
}


def txt_to_matrix(text):
    """Read a pose txt (16 numbers, row-major) as the [4, 4] camera-to-world matrix."""
    matrix = np.loadtxt(text)
    matrix = matrix.reshape(4, 4)
    return matrix


def get_cam_unit_vector(pose_matrix):
    """Unit vector from the origin to the camera. Input is the [4, 4] pose; the camera position is pose[:3, 3]."""
    t = pose_matrix[:3, 3]
    return t / np.linalg.norm(t)


def load_rgb(png):
    """png -> [3, 64, 64] float32 in [-1, 1]: composite on white, scale to [-1, 1], 2x2 box average."""
    im = Image.open(png).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))   # white background, composited before resizing
    rgb = Image.alpha_composite(bg, im).convert("RGB")
    x = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 127.5 - 1   # [3, 128, 128] in [-1, 1]
    return F.avg_pool2d(x, kernel_size=2)   # [3, 64, 64]


def view_names(obj):
    """View names (e.g. "000000") of an object folder that have both a pose txt and an rgb png."""
    poses = {txt.stem for txt in (obj / "pose").glob("*.txt")}
    rgbs = {png.stem for png in (obj / "rgb").glob("*.png")}
    return sorted(poses & rgbs)


class DLDataset(Dataset):
    """Reads the original folders: <path>/<object>/pose/<view>.txt and <path>/<object>/rgb/<view>.png."""

    def __init__(self, path, exclude=EXCLUDED_OBJECTS):
        path = Path(path)
        self.objects = sorted(p for p in path.glob("[!.]*") if p.is_dir() and p.name not in exclude)
        self.views = [view_names(obj) for obj in self.objects]   # poses and images are read lazily in __getitem__
        for obj, views in zip(self.objects, self.views):
            assert len(views) >= 2, f"{obj.name} has {len(views)} usable views, need 2 to make a pair"

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj = self.objects[idx]

        # Random pair: fine for training. Validation and test need fixed pairs.
        v1, v2 = random.sample(self.views[idx], 2)

        img1 = load_rgb(obj / "rgb" / f"{v1}.png")
        img2 = load_rgb(obj / "rgb" / f"{v2}.png")
        dir1 = get_cam_unit_vector(txt_to_matrix(obj / "pose" / f"{v1}.txt"))
        dir2 = get_cam_unit_vector(txt_to_matrix(obj / "pose" / f"{v2}.txt"))

        # source direction first, target direction second: this order is the model's pose conditioning
        d_pose = torch.from_numpy(np.concatenate([dir1, dir2])).float()
        return img1, img2, d_pose, obj.name


def packed_to_tensor(a):
    """[64, 64, 3] uint8 slice of a read-only memmap -> [3, 64, 64] float32 in [-1, 1]."""
    a = np.array(a)   # copy out of the memmap first: torch.from_numpy on a read-only array warns
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float() / 127.5 - 1


class PackedDataset(Dataset):
    """Same items as DLDataset, read from <packed_dir>/<split>_{images,directions}.npy and <split>_manifest.json."""

    def __init__(self, packed_dir, split="train"):
        packed_dir = Path(packed_dir)
        manifest = json.loads((packed_dir / f"{split}_manifest.json").read_text())
        self.images_file = packed_dir / manifest["images"]["file"]
        self.directions_file = packed_dir / manifest["directions"]["file"]
        # where each object came from; only .name is used here
        self.objects = [Path(manifest["source"]) / obj_id for obj_id in manifest["object_ids"]]
        if "view_names" in manifest:   # slot k of every object is view_names[k]
            names = manifest["view_names"]
            self.views = [names] * len(self.objects)   # one shared list, replaced (never modified) per object
            self.slots = [{v: k for k, v in enumerate(names)}] * len(self.objects)
        else:
            self.views = [manifest["view_names_per_object"][obj.name] for obj in self.objects]
            self.slots = [{v: k for k, v in enumerate(views)} for views in self.views]
        for obj, views in zip(self.objects, self.views):
            assert len(views) >= 2, f"{obj.name} has {len(views)} views, need 2 to make a pair"
        # opened on first use, so every DataLoader worker maps the files itself and shares them through the OS cache
        self._images = self._directions = None

    def __getstate__(self):
        # the memmaps must not travel to workers: pickling one would copy the whole array
        state = self.__dict__.copy()
        state["_images"] = state["_directions"] = None
        return state

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        if self._images is None:
            self._images = np.load(self.images_file, mmap_mode="r")
            self._directions = np.load(self.directions_file, mmap_mode="r")

        # Random pair: fine for training. Validation and test need fixed pairs.
        # Same call as DLDataset, so the same seed gives the same two views in both.
        v1, v2 = random.sample(self.views[idx], 2)
        k1, k2 = self.slots[idx][v1], self.slots[idx][v2]

        img1 = packed_to_tensor(self._images[idx, k1])
        img2 = packed_to_tensor(self._images[idx, k2])
        d_pose = torch.from_numpy(np.concatenate([self._directions[idx, k1], self._directions[idx, k2]]))
        return img1, img2, d_pose, self.objects[idx].name
