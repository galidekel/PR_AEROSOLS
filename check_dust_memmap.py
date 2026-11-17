"""
Check the dust input the model trains on (memmap dust.dat) for missing / empty cells, and compare it
with the raw CAMS files at the same time steps.

Why: the memmap regional mean is raw mean / 1.984 (constant) while the max is identical -> maybe ~half of
the grid cells are zero/empty. This script says whether that is true and WHERE the empty cells are.

Prints:
  1. dust.dat shape (expected H=61, W=63) and the time range
  2. NaN / exactly-zero fractions per channel
  3. distribution of the per-cell zero fraction (how many cells are ALWAYS zero, how many sometimes)
  4. fraction of time steps that are entirely zero
  5. (with --config) raw CAMS vs memmap at ~12 sampled time steps: grid shape, mean, max, ratio,
     and the fraction of cells where memmap == 0 but raw > 0
Saves: check_dust_memmap.png (maps of zero-fraction and mean, per channel)

Usage (HPC):
    python check_dust_memmap.py --memmap_dir /home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS_DATA/memmap --config config_hpc.yaml
    python check_dust_memmap.py --memmap_dir ... --no-raw        # skip the raw comparison (fast)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CH = ["total AOD550", "dust AOD550"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memmap_dir", required=True)
    p.add_argument("--config", default="config_hpc.yaml")
    p.add_argument("--no-raw", action="store_true", help="skip the comparison with the raw CAMS files")
    p.add_argument("--n_samples", type=int, default=12)
    p.add_argument("--out", default="check_dust_memmap.png")
    a = p.parse_args()

    d = Path(a.memmap_dir)
    meta = json.loads((d / "meta.json").read_text())
    T, C, H, W = meta["dust_shape"]
    times = pd.to_datetime(meta["times"])
    mm = np.memmap(d / "dust.dat", dtype="float32", mode="r", shape=(T, C, H, W))
    print(f"\n[1] dust.dat shape T={T} C={C} H={H} W={W}   {times.min()} -> {times.max()}")
    print(f"    expected H=61, W=63 for area_africa [35,-17,-10,30] at 0.75 deg")

    # One pass over the file: NaN/zero counts, per-cell zero count and sum, per-step all-zero flag.
    n_nan = np.zeros(C, dtype=np.int64); n_zero = np.zeros(C, dtype=np.int64)
    cell_zero = np.zeros((C, H, W), dtype=np.int64); cell_sum = np.zeros((C, H, W))
    step_allzero = np.zeros((T, C), dtype=bool)
    for s in range(0, T, 1000):
        x = np.asarray(mm[s:s + 1000])
        n_nan += np.isnan(x).sum(axis=(0, 2, 3))
        x = np.nan_to_num(x, nan=0.0)
        z = (x == 0)
        n_zero += z.sum(axis=(0, 2, 3))
        cell_zero += z.sum(axis=0)
        cell_sum += x.sum(axis=0)
        step_allzero[s:s + 1000] = z.all(axis=(2, 3))
    per_ch = T * H * W

    print("\n[2] NaN / zero cells per channel")
    for c in range(C):
        print(f"    {CH[c]:13s}: NaN {100 * n_nan[c] / per_ch:6.2f}%   exactly-zero (after NaN->0) {100 * n_zero[c] / per_ch:6.2f}%")

    print("\n[3] per-cell zero fraction over time (a cell that is zero at every step is 'always empty')")
    for c in range(C):
        zf = cell_zero[c] / T
        print(f"    {CH[c]:13s}: always-zero cells {100 * np.mean(zf >= 0.999):5.1f}%   "
              f"zero >50% of the time {100 * np.mean(zf > 0.5):5.1f}%   never zero {100 * np.mean(zf == 0):5.1f}%")

    print("\n[4] time steps that are entirely zero")
    for c in range(C):
        print(f"    {CH[c]:13s}: {int(step_allzero[:, c].sum())} of {T} steps ({100 * step_allzero[:, c].mean():.2f}%)")

    # ---- raw comparison -------------------------------------------------------------------------
    if not a.no_raw:
        from train_main import load_config, load_all_files, crop_area
        cfg = load_config(a.config)
        ds = load_all_files(Path(cfg["cams_dir"]), cfg["pattern_cams"], chunks={"time": cfg["chunk_time"]})
        ds = crop_area(ds, cfg["area_africa"])
        lat = "latitude" if "latitude" in ds.coords else "lat"
        lon = "longitude" if "longitude" in ds.coords else "lon"
        print(f"\n[5] raw CAMS after crop_area: lat={ds.sizes[lat]} lon={ds.sizes[lon]}   (memmap H={H} W={W})"
              f"{'   <-- SHAPE MISMATCH' if (ds.sizes[lat], ds.sizes[lon]) != (H, W) else '   shapes match'}")

        raw_t = pd.DatetimeIndex(ds.time.values)
        common = times[times.isin(raw_t)]
        print(f"    memmap steps found in raw CAMS: {len(common)} of {T}")
        pick = common[np.linspace(0, len(common) - 1, min(a.n_samples, len(common))).astype(int)]
        pos = {t: i for i, t in enumerate(times)}
        print(f"\n    {'time':20s} {'var':6s} {'raw mean':>9s} {'mm mean':>9s} {'ratio':>6s} {'raw max':>8s} {'mm max':>8s} "
              f"{'raw NaN%':>8s} {'mm==0&raw>0 %':>14s}")
        ratios = []
        for t in pick:
            for c, var in ((0, "aod550"), (1, "duaod550")):
                r = np.asarray(ds[var].sel(time=t).values, dtype=np.float32)
                m = np.asarray(mm[pos[t], c], dtype=np.float32); m = np.nan_to_num(m, nan=0.0)
                rn = np.isnan(r).mean() * 100
                if r.shape == m.shape:
                    mis = 100 * np.mean((m == 0) & (np.nan_to_num(r) > 0))
                else:
                    mis = float("nan")
                ratio = np.nanmean(r) / m.mean() if m.mean() > 0 else float("nan")
                ratios.append(ratio)
                print(f"    {str(t):20s} {var:6s} {np.nanmean(r):9.4f} {m.mean():9.4f} {ratio:6.2f} "
                      f"{np.nanmax(r):8.3f} {m.max():8.3f} {rn:8.1f} {mis:14.1f}")
        print(f"\n    median raw-mean / memmap-mean ratio over samples: {np.nanmedian(ratios):.2f}")

    # ---- maps -----------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, C, figsize=(5.5 * C, 9), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for c in range(C):
        im = axes[0, c].imshow(cell_zero[c] / T, vmin=0, vmax=1, cmap="magma_r", origin="upper")
        axes[0, c].set_title(f"{CH[c]}: fraction of time a cell is exactly 0")
        fig.colorbar(im, ax=axes[0, c], shrink=0.8)
        im = axes[1, c].imshow(cell_sum[c] / T, cmap="viridis", origin="upper")
        axes[1, c].set_title(f"{CH[c]}: mean over all time (memmap)")
        fig.colorbar(im, ax=axes[1, c], shrink=0.8)
    fig.savefig(a.out, dpi=130)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
