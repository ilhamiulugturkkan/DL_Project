import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

def txt_to_matrix(text):
    matrix = np.loadtxt(text)
    matrix = matrix.reshape(4,4)
    return matrix

def get_cam_unit_vector(pose_matrix): # input is pose txt's [4,4] matrix (transformed)
    t = pose_matrix[:3, 3]
    return t / np.linalg.norm(t)

def load_rgb(png):
    im = Image.open(png).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))   # white background
    rgb = Image.alpha_composite(bg, im).convert("RGB")
    # [3, H, W] in [-1, 1] AS RELATED WORK SCALES BETWEEN -1 AND 1
    x = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 127.5 - 1   # [3, 128, 128]
    return F.avg_pool2d(x, kernel_size=2)   # box / area average of each 2x2 block -> [3, 64, 64]

class DLDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        self.objects = sorted(p for p in path.glob("[!.]*") if p.is_dir())
        # view names (e.g. "000000") per object; poses/images are read lazily in __getitem__
        self.views = [sorted(txt.stem for txt in (obj / "pose").glob("*.txt")) for obj in self.objects]

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        # two distinct random views of the same object
        obj = self.objects[idx]

        # OK for training but needs fixed values for val and test
        v1, v2 = random.sample(self.views[idx], 2)

        img1 = load_rgb(obj / "rgb" / f"{v1}.png")
        img2 = load_rgb(obj / "rgb" / f"{v2}.png")
        dir1 = get_cam_unit_vector(txt_to_matrix(obj / "pose" / f"{v1}.txt"))
        dir2 = get_cam_unit_vector(txt_to_matrix(obj / "pose" / f"{v2}.txt"))


        # IMPORTANT!!!!!!!
        # d_pose: [6] = source cam unit direction (xyz), then target cam unit direction (xyz)
        d_pose = torch.from_numpy(np.concatenate([dir1, dir2])).float()

        # (src_img [3,H,W], tgt_img [3,H,W], d_pose [6], obj_id str); view 1 is source, view 2 is target
        return img1, img2, d_pose, obj.name

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="dataset directory, e.g. cars_train/")
    args = parser.parse_args()

    dl = DLDataset(path=args.root)

    loader = DataLoader(dl, batch_size=8, shuffle=True, num_workers=2)
    img1, img2, d_pose, obj_id = next(iter(loader))
    print(img1.shape, img2.shape, d_pose.shape, obj_id)