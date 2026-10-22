"""Load one batch through a DataLoader and print shapes and value range: the quickest "does the data pipeline work" check.

    python tools/smoke_batch.py --packed packed      # PackedDataset (fast)
    python tools/smoke_batch.py --root cars_train    # DLDataset (original files)
"""
import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from SRNDataset import DLDataset, PackedDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="dataset folder with one folder per object, e.g. cars_train/")
    parser.add_argument("--packed", help="folder written by pack_dataset.py, e.g. packed/ (split train)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    assert bool(args.root) != bool(args.packed), "give exactly one of --root and --packed"

    ds = PackedDataset(args.packed) if args.packed else DLDataset(args.root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    img1, img2, d_pose, obj_id = next(iter(loader))
    print(f"{type(ds).__name__}: {len(ds)} objects")
    print("src", tuple(img1.shape), "tgt", tuple(img2.shape), "d_pose", tuple(d_pose.shape), "obj_id", len(obj_id), "strings")
    print(f"src min {img1.min().item():.4f} max {img1.max().item():.4f} | tgt min {img2.min().item():.4f} max {img2.max().item():.4f}")


if __name__ == "__main__":   # the guard is needed for DataLoader workers (spawn)
    main()
