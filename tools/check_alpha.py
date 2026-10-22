"""Check the value ranges of the rgb pngs on a random sample (not the whole dataset).

    python tools/check_alpha.py [--n 300] [--save 6] [--out outputs/alpha_check]

Prints: modes and sizes, min / max alpha, whether alpha strictly between 0 and 255 exists, the (0, 0) corner pixel,
the raw RGB range and the range of load_rgb's output. Saves a few images (left: composited on white, right: the alpha
channel, 4x nearest-neighbour) so the edges can be looked at. On this data alpha is 255 everywhere, see docs/DECISIONS.md.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from SRNDataset import load_rgb

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=str(HERE / "cars_train"))
parser.add_argument("--n", type=int, default=300, help="random images to scan")
parser.add_argument("--save", type=int, default=6, help="how many composited images to write to --out")
parser.add_argument("--out", default=str(HERE / "outputs" / "alpha_check"))
args = parser.parse_args()

rng = random.Random(0)
objects = sorted(p for p in Path(args.root).glob("[!.]*") if p.is_dir())
pngs = []
for _ in range(args.n):
    obj = rng.choice(objects)
    pngs.append(obj / "rgb" / rng.choice(sorted(p.name for p in (obj / "rgb").glob("*.png"))))

modes, sizes, corners = Counter(), Counter(), Counter()
alpha_min, alpha_max = 255, 0
rgb_min, rgb_max = 255, 0
partial_imgs, partial_px, total_px = 0, 0, 0
out_min, out_max, out_shapes = 1.0, -1.0, set()

for png in pngs:
    im = Image.open(png)
    modes[im.mode] += 1
    sizes[im.size] += 1
    rgba = np.asarray(im.convert("RGBA"))   # an RGB png would get alpha 255 here, hence the mode count above
    alpha = rgba[..., 3]
    alpha_min, alpha_max = min(alpha_min, alpha.min()), max(alpha_max, alpha.max())
    rgb_min, rgb_max = min(rgb_min, rgba[..., :3].min()), max(rgb_max, rgba[..., :3].max())
    partial = (alpha > 0) & (alpha < 255)
    partial_imgs += partial.any()
    partial_px += partial.sum()
    total_px += alpha.size
    corners[tuple(int(v) for v in rgba[0, 0])] += 1

    x = load_rgb(png)
    out_min, out_max = min(out_min, x.min().item()), max(out_max, x.max().item())
    out_shapes.add(tuple(x.shape))

print(f"scanned {len(pngs)} images from {len(set(p.parent.parent for p in pngs))} objects")
print("modes:", dict(modes), "sizes:", dict(sizes))
print(f"alpha: min {alpha_min}, max {alpha_max}")
print(f"alpha strictly between 0 and 255: {'yes' if partial_px else 'no'} "
      f"({partial_imgs}/{len(pngs)} images, {100 * partial_px / total_px:.2f}% of pixels)")
print("corner pixel (0, 0) RGBA:", dict(corners))
print(f"raw rgb: min {rgb_min}, max {rgb_max}")
print(f"load_rgb output: min {out_min:.4f}, max {out_max:.4f}, shapes {sorted(out_shapes)}")

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for png in pngs[:args.save]:
    im = Image.open(png).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))   # same compositing as load_rgb
    rgb = Image.alpha_composite(bg, im).convert("RGB")
    alpha = im.getchannel("A").convert("RGB")
    both = Image.new("RGB", (2 * im.width, im.height))
    both.paste(rgb, (0, 0))
    both.paste(alpha, (im.width, 0))
    both = both.resize((both.width * 4, both.height * 4), Image.NEAREST)   # nearest, so single pixels stay visible
    both.save(out / f"{png.parent.parent.name[:8]}_{png.stem}.png")
print(f"saved {min(args.save, len(pngs))} composites (left: on white, right: alpha) to {out}/")
