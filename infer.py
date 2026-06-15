"""
Inference on the val split using a saved checkpoint.

Usage:
    python infer.py --checkpoint checkpoints/best.pt --config config_hpc.yaml
    python infer.py --checkpoint checkpoints/best.pt --config config_hpc.yaml --n_batches 20
    python infer.py --checkpoint checkpoints/best.pt --config config_hpc.yaml --out results.csv
"""

import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pr_VIT_dustmeteo import PRAfricaDustRouteMeteoNet
from train_main import (
    load_config, load_all_files, load_norm_stats,
    crop_area, build_route_dataset, build_dust_dataset,
    load_aeronet_targets, PRLazyDataset,
)
from pathlib import Path


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config",     default="config_hpc.yaml")
    parser.add_argument("--n_batches",  type=int, default=0,
                        help="Number of batches to run (0 = all val set)")
    parser.add_argument("--out",        default="infer_results.csv",
                        help="Path to save predictions CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    route_channel_names = ckpt["route_channel_names"]
    log(f"ckpt epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}")

    # ── Data pipeline (mirrors train_main.py exactly) ─────────────────────────
    ERA5_DIR = Path(cfg["era5_dir"])
    CAMS_DIR = Path(cfg["cams_dir"])
    chunks   = {"time": cfg["chunk_time"]}
    T_STEP   = cfg["t_step_hours"]

    log("loading ERA5 PL ...")
    ds_pl = load_all_files(ERA5_DIR, cfg["pattern_era5_pl"], chunks=chunks)
    log("loading ERA5 SL ...")
    ds_sl = load_all_files(ERA5_DIR, cfg["pattern_era5_sl"], chunks=chunks)
    log("loading CAMS ...")
    ds_cams = load_all_files(CAMS_DIR, cfg["pattern_cams"], chunks=chunks)

    ds_pl_route    = crop_area(ds_pl,   cfg["area_route"])
    ds_sl_route    = crop_area(ds_sl,   cfg["area_route"])
    ds_cams_africa = crop_area(ds_cams, cfg["area_africa"])

    route_da = build_route_dataset(ds_pl_route, ds_sl_route)
    dust_da  = build_dust_dataset(ds_cams_africa)

    _, py, px = tuple(cfg["route_patch_size"])
    H_trim = (route_da.shape[2] // py) * py
    W_trim = (route_da.shape[3] // px) * px
    if H_trim != route_da.shape[2] or W_trim != route_da.shape[3]:
        route_da = route_da.isel(latitude=slice(0, H_trim), longitude=slice(0, W_trim))

    all_times    = pd.DatetimeIndex(route_da.time.values)
    target_times = all_times[all_times.hour % T_STEP == 0]
    route_da = route_da.sel(time=target_times)
    dust_da  = dust_da.sel(time=target_times)
    times    = route_da.time.values

    targets = load_aeronet_targets(cfg["aeronet_path"])

    route_mean, route_std, dust_mean, dust_std = load_norm_stats(cfg["norm_stats_path"])

    full_dataset = PRLazyDataset(
        route_da, dust_da, targets, times,
        cfg["t_in"], cfg["t_dust"],
        route_mean, route_std, dust_mean, dust_std,
        memmap_dir=cfg.get("memmap_dir"),
    )

    n_val   = max(1, int(len(full_dataset) * cfg["val_frac"]))
    n_train = len(full_dataset) - n_val
    val_ds  = Subset(full_dataset, range(n_train, len(full_dataset)))
    log(f"split: total={len(full_dataset)}  train={n_train}  val={len(val_ds)}")

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── Build model and load weights ───────────────────────────────────────────
    model = PRAfricaDustRouteMeteoNet(
        route_channel_names=route_channel_names,
        route_patch_size=tuple(cfg["route_patch_size"]),
        embed_dim=cfg["embed_dim"],
        num_heads_space=cfg["num_heads_space"],
        num_heads_fusion=cfg["num_heads_fusion"],
        dropout=cfg["dropout"],
        out_dim=cfg["out_dim"],
        T_in=cfg["t_in"],
        H_route=route_da.shape[2],
        W_route=route_da.shape[3],
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if unexpected:
        log(f"ignored keys not in model: {unexpected}")
    if missing:
        log(f"missing keys (will use random init): {missing}")
    model.eval()
    log("model loaded — starting inference")

    # ── Collect dates for each val sample ─────────────────────────────────────
    T_IN = cfg["t_in"]
    val_dates = [
        pd.Timestamp(times[full_dataset.samples[n_train + k][0] + T_IN - 1]).normalize()
        for k in range(len(val_ds))
    ]

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_targets, all_dates = [], [], []
    limit = args.n_batches if args.n_batches > 0 else len(val_loader)

    print(f"\n{'Batch':>5}  {'pred_min':>9} {'pred_max':>9} {'pred_mean':>10} "
          f"{'tgt_min':>8} {'tgt_max':>8} {'tgt_mean':>9}  {'MSE':>8}")
    print("-" * 80)

    with torch.no_grad():
        for batch_idx, (x_route, x_dust, y) in enumerate(val_loader):
            if batch_idx >= limit:
                break

            x_route = x_route.to(device)
            x_dust  = x_dust.to(device)

            y_hat = model(x_dust, x_route, return_attn=False)
            preds = y_hat.squeeze(-1).cpu().numpy()
            tgts  = y.numpy()

            start = batch_idx * cfg["batch_size"]
            all_dates.extend(val_dates[start : start + len(preds)])

            mse = float(((preds - tgts) ** 2).mean())
            print(f"{batch_idx+1:>5}  "
                  f"{preds.min():>9.4f} {preds.max():>9.4f} {preds.mean():>10.4f} "
                  f"{tgts.min():>8.4f} {tgts.max():>8.4f} {tgts.mean():>9.4f}  "
                  f"{mse:>8.4f}")

            all_preds.append(preds)
            all_targets.append(tgts)

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    overall_mse  = float(((all_preds - all_targets) ** 2).mean())
    baseline_mse = float(((all_targets - all_targets.mean()) ** 2).mean())
    r2 = 1.0 - overall_mse / (baseline_mse + 1e-9)

    log(f"inference done — {len(all_preds)} samples")
    print("-" * 80)
    log(f"pred  min={all_preds.min():.4f}  max={all_preds.max():.4f}  "
        f"mean={all_preds.mean():.4f}  std={all_preds.std():.4f}")
    log(f"tgt   min={all_targets.min():.4f}  max={all_targets.max():.4f}  "
        f"mean={all_targets.mean():.4f}  std={all_targets.std():.4f}")
    log(f"MSE={overall_mse:.4f}  baseline_MSE={baseline_mse:.4f}  R²={r2:.4f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    pd.DataFrame({
        "date":   [d.date() for d in all_dates],
        "pred":   all_preds,
        "target": all_targets,
    }).to_csv(out_path, index=False)
    log(f"saved {out_path}  ({len(all_preds)} rows)")


if __name__ == "__main__":
    main()
