"""How much time the packed arrays save, on a subset of objects: the old loader (DLDataset, png + txt files) against the
new one (PackedDataset, packed arrays), same objects, same order, through a DataLoader.

    python tools/time_save_test.py --subset 256 --batch-size 32 --workers 0 4 --step-time 0.25

--step-time is E's time per training step in seconds; with it the output says whether the loader delivers a batch
faster than the step consumes it. First pass = first epoch, includes worker start-up and cold files; steady = median of
the later epochs (files in the OS cache). Run on an otherwise idle machine.
"""
import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from SRNDataset import DLDataset, PackedDataset

HERE = Path(__file__).resolve().parent.parent

def per_sample_ms(ds, indices):
    # one process, no DataLoader: the cost of a single __getitem__
    ds[indices[0]]   # PackedDataset opens its memmaps on first use, not part of the per-sample cost
    random.seed(0)
    times = []
    for i in indices:
        t = time.perf_counter()
        ds[i]
        times.append((time.perf_counter() - t) * 1000)
    return statistics.mean(times), statistics.median(times)

def epoch_times(ds, indices, batch_size, workers, epochs):
    loader = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, generator=torch.Generator().manual_seed(0))
    times = []
    for _ in range(1 + epochs):
        n = 0
        t = time.perf_counter()
        for batch in loader:
            n += 1
        times.append(time.perf_counter() - t)
        assert batch[0].shape == (batch_size, 3, 64, 64) and batch[2].shape == (batch_size, 6)
    del loader
    return n, times[0], statistics.median(times[1:])   # batches per epoch, first pass, steady

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(HERE / "cars_train"))
    parser.add_argument("--packed", default=str(HERE / "packed"))
    parser.add_argument("--subset", type=int, default=256, help="number of objects used (same ones for both loaders)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--epochs", type=int, default=3, help="epochs after the first pass, for the steady number")
    parser.add_argument("--step-time", type=float, default=None, help="E's seconds per training step, to compare with")
    args = parser.parse_args()

    raw, packed = DLDataset(args.root), PackedDataset(args.packed)   # raw lists the folders (names only) here, no image is read
    assert [o.name for o in raw.objects] == [o.name for o in packed.objects], "datasets list different objects"
    assert args.subset >= args.batch_size, "subset smaller than one batch"
    indices = sorted(random.Random(0).sample(range(len(raw)), args.subset))
    print(f"{args.subset} of {len(raw)} objects, batch size {args.batch_size}, {args.epochs} steady epochs after the first pass\n")

    print("one __getitem__, single process (ms):")
    latency = {}
    for name, ds in (("old", raw), ("new", packed)):
        latency[name] = per_sample_ms(ds, indices[:min(200, len(indices))])
        print(f"  {name}: mean {latency[name][0]:7.3f}   median {latency[name][1]:7.3f}")
    print(f"  -> {latency['old'][0] / latency['new'][0]:.0f}x faster per sample (mean)\n")

    print("through a DataLoader (shuffle, drop_last):")
    header = f"  {'workers':>7} {'loader':>6} {'first pass s':>12} {'steady epoch s':>14} {'samples/s':>10} {'ms/batch':>9}"
    print(header)
    results = []
    for w in args.workers:
        row = {}
        for name, ds in (("old", raw), ("new", packed)):
            batches, first, steady = epoch_times(ds, indices, args.batch_size, w, args.epochs)
            row[name] = dict(batches=batches, first=first, steady=steady, sps=batches * args.batch_size / steady, ms=steady / batches * 1000)
            r = row[name]
            print(f"  {w:>7} {name:>6} {r['first']:>12.2f} {r['steady']:>14.2f} {r['sps']:>10.0f} {r['ms']:>9.2f}")
        results.append((w, row))

    print("\ntime saved:")
    for w, row in results:
        old, new = row["old"], row["new"]
        # ms per batch x 1000 batches / 1000 ms per s: the same number, in seconds per 1000 batches
        print(f"  {w} workers: {new['sps'] / old['sps']:.1f}x more samples/s; "
              f"1000 batches take {old['ms']:.1f} s -> {new['ms']:.1f} s (saves {old['ms'] - new['ms']:.1f} s); "
              f"steady epoch of these {args.subset} objects {old['steady']:.2f} s -> {new['steady']:.2f} s; "
              f"first pass {old['first']:.2f} s -> {new['first']:.2f} s")
        print(f"             a full pass over all {len(raw)} objects (extrapolated): {len(raw) / old['sps']:.1f} s -> {len(raw) / new['sps']:.1f} s")

    if args.step_time is not None:
        print(f"\nversus E's training step ({args.step_time * 1000:.0f} ms per step):")
        for w, row in results:
            for name in ("old", "new"):
                ms = row[name]["ms"]
                verdict = "loader is faster than the step, not the bottleneck" if ms < args.step_time * 1000 else "LOADER IS SLOWER than the step, the GPU waits"
                print(f"  {w} workers, {name}: {ms:.1f} ms per batch -> {verdict} (step / loader = {args.step_time * 1000 / ms:.1f}x)")
    else:
        print("\n(pass --step-time SECONDS with E's time per training step to compare it with the ms/batch above)")

if __name__ == "__main__":   # also needed for DataLoader workers (spawn)
    main()
