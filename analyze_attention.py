"""
Extract and aggregate attention weights from the two attention gates in
PRAfricaDustRouteMeteoNet, on the val split, using a saved checkpoint.

Gate 1 (per-channel space-time gate, inside RouteMeteoEncoder):
    one learnable token per meteo channel attends over that channel's
    space-time patches -> shows WHERE in the domain / WHEN in the T_in-day
    window the model focuses, per variable.

Gate 2 (fusion gate, pr_to_fused):
    the PR token attends over [dust_source_summary, route_channel_summaries...]
    -> shows WHICH source/variable matters most for the final prediction.

This mirrors infer.py's data pipeline exactly, but calls the model with
return_attn=True and aggregates the attention maps instead of (only) MSE.
Heavy (loads full ERA5/CAMS) -- run this on the HPC the same way you run
infer.py, then scp the two output files to your local machine and plot
with plot_attention.py.

Usage:
    python analyze_attention.py --checkpoint outputs/run_3_stations_/best.pt \
        --config config_hpc.yaml --out_dir outputs/run_3_stations_/attention
    python analyze_attention.py --checkpoint ... --config ... --n_batches 20   # quick test
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pr_VIT_dustmeteo import PRAfricaDustRouteMeteoNet
from train_main import (
    load_config, load_all_files, load_norm_stats,
    crop_area, build_route_dataset, build_dust_dataset,
    load_station_targets, PRLazyDataset,
)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config",     default="config_hpc.yaml")
    parser.add_argument("--n_batches",  type=int, default=0,
                        help="Number of val batches to run (0 = all)")
    parser.add_argument("--out_dir",    default="attention_out",
                        help="Directory to save fusion_weights.csv and route_attention.npz")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    # ── Load checkpoint (mirrors infer.py) ─────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    route_channel_names = ckpt["route_channel_names"]
    inference_station = ckpt.get("inference_station", cfg["inference_station"])
    station_names = ckpt.get("station_names", [inference_station])
    log(f"ckpt epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f}  "
        f"inference_station={inference_station!r}")

    primary_cfg = next(s for s in cfg["stations"] if s["name"] == inference_station)

    # ── Data pipeline (identical to infer.py / train_main.py) ─────────────
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

    targets = load_station_targets(primary_cfg, default_agg=cfg.get("target_agg", "median"))

    route_mean, route_std, dust_mean, dust_std = load_norm_stats(cfg["norm_stats_path"])

    full_dataset = PRLazyDataset(
        route_da, dust_da, targets, times,
        cfg["t_in"], cfg["t_dust"],
        route_mean, route_std, dust_mean, dust_std,
        memmap_dir=cfg.get("memmap_dir"),
        station_name=inference_station,
    )

    n_val   = max(1, int(len(full_dataset) * cfg["val_frac"]))
    n_train = len(full_dataset) - n_val
    val_ds  = Subset(full_dataset, range(n_train, len(full_dataset)))
    log(f"split: total={len(full_dataset)}  train={n_train}  val={len(val_ds)}")

    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── Build model and load weights ───────────────────────────────────────
    model = PRAfricaDustRouteMeteoNet(
        route_channel_names=route_channel_names,
        station_names=station_names,
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
    log("model loaded — starting attention extraction")

    # ── Collect dates for each val sample (mirrors infer.py) ──────────────
    T_IN = cfg["t_in"]
    val_dates = [
        pd.Timestamp(times[full_dataset.samples[n_train + k][0] + T_IN - 1]).normalize()
        for k in range(len(val_ds))
    ]

    fusion_token_names = None
    fusion_rows = []          # one row per sample: date, pred, target, <token weights...>

    # Running sums for the Gate-1 (per-channel space-time) attention, so we
    # never hold [N, Tp, Hp, Wp] per channel in memory at once — only the
    # aggregated mean pattern, which is what's actually useful to plot.
    spatial_sum  = {}   # ch -> [Hp, Wp] running sum (mean over T per sample, summed over samples)
    temporal_sum = {}   # ch -> [Tp]    running sum (mean over H,W per sample, summed over samples)
    n_seen = 0
    Tp = Hp = Wp = None

    limit = args.n_batches if args.n_batches > 0 else len(val_loader)

    with torch.no_grad():
        for batch_idx, (x_route, x_dust, y) in enumerate(val_loader):
            if batch_idx >= limit:
                break

            x_route = x_route.to(device)
            x_dust  = x_dust.to(device)

            y_hat, attn_route, attn_fusion, meta = model(
                x_dust, x_route, station=inference_station, return_attn=True
            )
            preds = y_hat.squeeze(-1).cpu().numpy()
            tgts  = y.numpy()
            bs = len(preds)

            if fusion_token_names is None:
                fusion_token_names = meta["fusion_token_names"]
                Tp, Hp, Wp = meta["meta_route"]["Tp"], meta["meta_route"]["Hp"], meta["meta_route"]["Wp"]
                for ch in route_channel_names:
                    spatial_sum[ch]  = np.zeros((Hp, Wp), dtype=np.float64)
                    temporal_sum[ch] = np.zeros((Tp,), dtype=np.float64)
                log(f"Gate 1 grid: Tp={Tp} Hp={Hp} Wp={Wp}  |  Gate 2 tokens: {fusion_token_names}")

            # -- Gate 2: fusion weights, kept per-sample (small: N x (1+C_r)) --
            fw = attn_fusion.squeeze(1).cpu().numpy()   # [B, 1+C_r]
            start = batch_idx * cfg["batch_size"]
            for k in range(bs):
                row = {
                    "date":   val_dates[start + k].date(),
                    "pred":   float(preds[k]),
                    "target": float(tgts[k]),
                }
                row.update({name: float(fw[k, i]) for i, name in enumerate(fusion_token_names)})
                fusion_rows.append(row)

            # -- Gate 1: per-channel space-time attention, accumulated as running mean --
            for ch, a in attn_route.items():
                a = a.squeeze(1).cpu().numpy()          # [B, Tp, Hp, Wp]
                spatial_sum[ch]  += a.mean(axis=1).sum(axis=0)          # mean over T, sum over batch
                temporal_sum[ch] += a.mean(axis=(2, 3)).sum(axis=0)     # mean over H,W, sum over batch

            n_seen += bs
            log(f"batch {batch_idx + 1}/{limit}  ({n_seen} samples so far)")

    # ── Save Gate 2 (fusion) weights — per-sample CSV ──────────────────────
    fusion_df = pd.DataFrame(fusion_rows)
    fusion_csv = out_dir / "fusion_weights.csv"
    fusion_df.to_csv(fusion_csv, index=False)
    log(f"saved {fusion_csv}  ({len(fusion_df)} rows)")

    mean_importance = fusion_df[fusion_token_names].mean().sort_values(ascending=False)
    log("Gate 2 mean attention weight per token (sorted):")
    for name, val in mean_importance.items():
        log(f"    {name:25s} {val:.4f}")

    # ── Save Gate 1 (per-channel space-time) aggregated maps — npz ─────────
    npz_path = out_dir / "route_attention.npz"
    save_dict = {
        "channel_names": np.array(route_channel_names),
        "Tp": Tp, "Hp": Hp, "Wp": Wp,
        "n_samples": n_seen,
        "route_patch_size": np.array(cfg["route_patch_size"]),
        "area_route": np.array(cfg["area_route"]),
        "t_step_hours": cfg["t_step_hours"],
        "t_in_days": cfg["t_in_days"],
    }
    for ch in route_channel_names:
        save_dict[f"spatial__{ch}"]  = (spatial_sum[ch] / n_seen).astype(np.float32)
        save_dict[f"temporal__{ch}"] = (temporal_sum[ch] / n_seen).astype(np.float32)
    np.savez(npz_path, **save_dict)
    log(f"saved {npz_path}  ({len(route_channel_names)} channels, averaged over {n_seen} samples)")

    log("done")


if __name__ == "__main__":
    main()
