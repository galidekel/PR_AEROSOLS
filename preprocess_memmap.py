"""
One-time preprocessing: writes route and dust xarray DataArrays to flat
float32 binary files on scratch for fast numpy memmap access during training.

Run once before training:
    python preprocess_memmap.py --config config_hpc.yaml

Outputs:
    $SCRATCH/route.dat  — float32 [T, C_r, H, W]
    $SCRATCH/dust.dat   — float32 [T, C_d, H_d, W_d]
    $SCRATCH/meta.json  — shapes and dtype info
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_main import (
    load_config, load_all_files, crop_area,
    build_route_dataset, build_dust_dataset,
)

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_hpc.yaml")
    parser.add_argument("--scratch", default="/scratch/galidek_pr")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ERA5_DIR = cfg["era5_dir"]
    CAMS_DIR = cfg["cams_dir"]
    T_STEP_HOURS = cfg["t_step_hours"]
    ROUTE_PATCH_SIZE = tuple(cfg["route_patch_size"])
    AREA_ROUTE  = list(cfg["area_route"])
    AREA_AFRICA = list(cfg["area_africa"])
    SCRATCH = Path(args.scratch)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    chunks = {"time": cfg["chunk_time"]}

    print("--- LOADING ERA5 PL ---")
    ds_pl = load_all_files(ERA5_DIR, cfg["pattern_era5_pl"], chunks=chunks)
    print("--- LOADING ERA5 SL ---")
    ds_sl = load_all_files(ERA5_DIR, cfg["pattern_era5_sl"], chunks=chunks)
    print("--- LOADING CAMS ---")
    ds_cams = load_all_files(CAMS_DIR, cfg["pattern_cams"], chunks=chunks)

    ds_pl_route    = crop_area(ds_pl,   AREA_ROUTE)
    ds_sl_route    = crop_area(ds_sl,   AREA_ROUTE)
    ds_cams_africa = crop_area(ds_cams, AREA_AFRICA)

    route_da = build_route_dataset(ds_pl_route, ds_sl_route)
    dust_da  = build_dust_dataset(ds_cams_africa)

    _, py, px = ROUTE_PATCH_SIZE
    H_trim = (route_da.shape[2] // py) * py
    W_trim = (route_da.shape[3] // px) * px
    if H_trim != route_da.shape[2] or W_trim != route_da.shape[3]:
        route_da = route_da.isel(latitude=slice(0, H_trim), longitude=slice(0, W_trim))
        print(f"[spatial] route trimmed to {H_trim}×{W_trim}")

    all_times    = pd.DatetimeIndex(route_da.time.values)
    target_times = all_times[all_times.hour % T_STEP_HOURS == 0]
    route_da = route_da.sel(time=target_times)
    dust_da  = dust_da.sel(time=target_times)

    T, C_r, H, W    = route_da.shape
    T2, C_d, Hd, Wd = dust_da.shape
    assert T == T2

    print(f"\nRoute shape: {route_da.shape}  → {T * C_r * H * W * 4 / 1e9:.1f} GB")
    print(f"Dust  shape: {dust_da.shape}   → {T * C_d * Hd * Wd * 4 / 1e9:.3f} GB")

    route_path = SCRATCH / "route.dat"
    dust_path  = SCRATCH / "dust.dat"
    meta_path  = SCRATCH / "meta.json"

    # Write route in time chunks to avoid loading everything at once
    WRITE_CHUNK = 64
    route_mm = np.memmap(route_path, dtype="float32", mode="w+",
                         shape=(T, C_r, H, W))
    print(f"\nWriting route → {route_path}")
    for start in tqdm(range(0, T, WRITE_CHUNK), unit="chunk"):
        end = min(start + WRITE_CHUNK, T)
        route_mm[start:end] = route_da.isel(time=slice(start, end)).values.astype(np.float32)
    route_mm.flush()
    del route_mm
    print("Route written.")

    # Dust is tiny — load all at once
    print(f"\nWriting dust  → {dust_path}")
    dust_mm = np.memmap(dust_path, dtype="float32", mode="w+",
                        shape=(T, C_d, Hd, Wd))
    dust_mm[:] = dust_da.values.astype(np.float32)
    dust_mm.flush()
    del dust_mm
    print("Dust written.")

    meta = {
        "route_shape": [T, C_r, H, W],
        "dust_shape":  [T, C_d, Hd, Wd],
        "times": [str(t) for t in target_times],
        "route_channel_names": list(route_da.coords["channel"].values),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"\nMeta saved → {meta_path}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
