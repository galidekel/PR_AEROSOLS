"""
Channel ablation: does the model's prediction actually DEPEND on a channel (e.g. pv_700)?

Attention weights only say where the model reads. This measures dependence directly: on the val
split, take the trained checkpoint, replace one channel (or channel group) with uninformative
values, and see how much the error rises.

  --mode shuffle (default)  replace the channel with the SAME channel from a DIFFERENT, randomly
                            chosen val day (keeps realistic marginal statistics, destroys the link
                            to this day's AOD). Repeated --n_repeats times with different pairings.
  --mode zero               replace the channel with 0 = the training mean (inputs are z-scored).

Specs for --channels:  pv_700        one route channel
                       pv_*          every channel whose name starts with 'pv_'   (a group)
                       dust          the CAMS dust-source branch (x_dust)

Reports val MSE for the intact model, MSE after each ablation, the increase (dMSE) with a paired
bootstrap 95% CI over val days, and a climatology baseline (training-mean AOD) for scale.
A dMSE whose CI excludes 0 and is comparable to the baseline-vs-climatology gap means the model
genuinely relies on that input; dMSE ~ 0 means the attention was incidental.

Heavy (loads full ERA5/CAMS) -- run on the HPC the same way as analyze_attention.py.

Usage:
    python ablate_channels.py --checkpoint outputs/run_3_stations_/best.pt --config config_hpc.yaml \
        --out_dir outputs/run_3_stations_/ablation
    python ablate_channels.py ... --channels pv_700 "pv_*" t_850 dust --n_repeats 5
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from pr_VIT_dustmeteo import PRAfricaDustRouteMeteoNet
from train_main import (
    load_config, load_all_files, load_norm_stats,
    crop_area, build_route_dataset, build_dust_dataset,
    load_station_targets, PRLazyDataset,
)

DEFAULT_SPECS = ["pv_700", "pv_900", "pv_850", "pv_*", "t_700", "t_850", "t_500", "z_500", "dust"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="config_hpc.yaml")
    parser.add_argument("--out_dir", default="ablation_out")
    parser.add_argument("--channels", nargs="*", default=DEFAULT_SPECS)
    parser.add_argument("--mode", choices=["shuffle", "zero"], default="shuffle")
    parser.add_argument("--n_repeats", type=int, default=3, help="Shuffle pairings to average (shuffle mode)")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    # ── Checkpoint + data pipeline: identical to analyze_attention.py ─────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    route_channel_names = list(ckpt["route_channel_names"])
    inference_station = ckpt.get("inference_station", cfg["inference_station"])
    station_names = ckpt.get("station_names", [inference_station])
    primary_cfg = next(s for s in cfg["stations"] if s["name"] == inference_station)

    ERA5_DIR, CAMS_DIR = Path(cfg["era5_dir"]), Path(cfg["cams_dir"])
    chunks, T_STEP = {"time": cfg["chunk_time"]}, cfg["t_step_hours"]
    ds_pl = load_all_files(ERA5_DIR, cfg["pattern_era5_pl"], chunks=chunks)
    ds_sl = load_all_files(ERA5_DIR, cfg["pattern_era5_sl"], chunks=chunks)
    ds_cams = load_all_files(CAMS_DIR, cfg["pattern_cams"], chunks=chunks)
    route_da = build_route_dataset(crop_area(ds_pl, cfg["area_route"]), crop_area(ds_sl, cfg["area_route"]))
    dust_da = build_dust_dataset(crop_area(ds_cams, cfg["area_africa"]))

    _, py, px = tuple(cfg["route_patch_size"])
    H_trim, W_trim = (route_da.shape[2] // py) * py, (route_da.shape[3] // px) * px
    if H_trim != route_da.shape[2] or W_trim != route_da.shape[3]:
        route_da = route_da.isel(latitude=slice(0, H_trim), longitude=slice(0, W_trim))
    all_times = pd.DatetimeIndex(route_da.time.values)
    target_times = all_times[all_times.hour % T_STEP == 0]
    route_da, dust_da = route_da.sel(time=target_times), dust_da.sel(time=target_times)
    times = route_da.time.values

    targets = load_station_targets(primary_cfg, default_agg=cfg.get("target_agg", "median"))
    route_mean, route_std, dust_mean, dust_std = load_norm_stats(cfg["norm_stats_path"])
    full = PRLazyDataset(route_da, dust_da, targets, times, cfg["t_in"], cfg["t_dust"],
                         route_mean, route_std, dust_mean, dust_std,
                         memmap_dir=cfg.get("memmap_dir"), station_name=inference_station)
    n_val = max(1, int(len(full) * cfg["val_frac"]))
    n_train = len(full) - n_val
    val_ds = Subset(full, range(n_train, len(full)))
    log(f"split: total={len(full)} train={n_train} val={n_val}")

    model = PRAfricaDustRouteMeteoNet(
        route_channel_names=route_channel_names, station_names=station_names,
        route_patch_size=tuple(cfg["route_patch_size"]), embed_dim=cfg["embed_dim"],
        num_heads_space=cfg["num_heads_space"], num_heads_fusion=cfg["num_heads_fusion"],
        dropout=cfg["dropout"], out_dim=cfg["out_dim"], T_in=cfg["t_in"],
        H_route=route_da.shape[2], W_route=route_da.shape[3],
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # ── Resolve specs -> channel index lists ────────────────────────────────
    specs = {}
    for s in args.channels:
        if s == "dust":
            specs[s] = "dust"
        elif s.endswith("*"):
            idx = [i for i, n in enumerate(route_channel_names) if n.startswith(s[:-1])]
            if idx:
                specs[s] = idx
            else:
                log(f"skipping {s!r}: no matching channels")
        elif s in route_channel_names:
            specs[s] = [route_channel_names.index(s)]
        else:
            log(f"skipping {s!r}: not a route channel")
    log(f"ablating: { {k: (v if v == 'dust' else len(v)) for k, v in specs.items()} }  (mode={args.mode})")

    rng = np.random.default_rng(args.seed)
    n_rep = args.n_repeats if args.mode == "shuffle" else 1
    perms = [rng.permutation(n_val) for _ in range(n_rep)]

    B = cfg["batch_size"]
    y_all = np.array([full.samples[n_train + i][1] for i in range(n_val)])   # targets only; no data loading
    se_base = np.zeros(n_val)
    se_abl = {k: np.zeros((n_rep, n_val)) for k in specs}

    @torch.no_grad()
    def predict(xd, xr):
        return model(xd, xr, station=inference_station, return_attn=False).squeeze(-1).cpu().numpy()

    for start in range(0, n_val, B):
        ids = list(range(start, min(start + B, n_val)))
        xr = torch.stack([val_ds[i][0] for i in ids]).to(device)
        xd = torch.stack([val_ds[i][1] for i in ids]).to(device)
        y = y_all[ids]
        se_base[ids] = (predict(xd, xr) - y) ** 2

        for r in range(n_rep):
            if args.mode == "shuffle":
                donors = [int(perms[r][i]) for i in ids]
                dr = torch.stack([val_ds[j][0] for j in donors]).to(device)
                dd = torch.stack([val_ds[j][1] for j in donors]).to(device)
            for k, v in specs.items():
                if v == "dust":
                    xd_m = dd if args.mode == "shuffle" else torch.zeros_like(xd)
                    pred = predict(xd_m, xr)
                else:
                    xr_m = xr.clone()                           # [B, T_in, C_r, H, W]
                    xr_m[:, :, v] = dr[:, :, v] if args.mode == "shuffle" else 0.0
                    pred = predict(xd, xr_m)
                se_abl[k][r, ids] = (pred - y) ** 2
        log(f"batch {start // B + 1}/{int(np.ceil(n_val / B))}")

    # ── Statistics ──────────────────────────────────────────────────────────
    train_mean = float(np.mean([full.samples[i][1] for i in range(n_train)]))
    mse_clim = float(np.mean((y_all - train_mean) ** 2))
    mse_base = float(se_base.mean())
    log(f"val MSE intact={mse_base:.6f}   climatology (train-mean AOD)={mse_clim:.6f}   "
        f"skill vs clim={1 - mse_base / mse_clim:.3f}")

    boot_rng = np.random.default_rng(args.seed + 1)
    boot_idx = boot_rng.integers(0, n_val, size=(args.n_boot, n_val))
    rows = []
    for k in specs:
        d_i = se_abl[k].mean(axis=0) - se_base                  # paired per-day increase
        boots = d_i[boot_idx].mean(axis=1)
        rows.append({"ablation": k, "mse": mse_base + d_i.mean(), "dMSE": d_i.mean(),
                     "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                     "pct_of_clim_gap": 100 * d_i.mean() / max(mse_clim - mse_base, 1e-12)})
    res = pd.DataFrame(rows).sort_values("dMSE", ascending=False)
    res.to_csv(out_dir / "ablation.csv", index=False)
    log("\n" + res.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    with open(out_dir / "ablation_summary.txt", "w") as f:
        f.write(f"mode={args.mode} repeats={n_rep} val_days={n_val}\n"
                f"MSE intact={mse_base:.6f}  climatology={mse_clim:.6f}\n\n"
                + res.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # ── Plot: increase in MSE per ablation, with bootstrap CI ───────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(r) + 2.2))
    ax.barh(r["ablation"], r["dMSE"], color="#2a78d6",
            xerr=[r["dMSE"] - r["ci_lo"], r["ci_hi"] - r["dMSE"]], capsize=3,
            error_kw={"ecolor": "#52514e", "elinewidth": 1.0})
    ax.axvline(0, color="#0b0b0b", linewidth=1)
    ax.axvline(mse_clim - mse_base, color="#eb6834", linestyle="--", linewidth=1.5,
               label="gap: climatology MSE - model MSE")
    ax.set_xlabel("Increase in val MSE when the input is removed (95% bootstrap CI over days)")
    ax.set_title(f"Channel ablation ({args.mode}); intact MSE={mse_base:.4f}, climatology={mse_clim:.4f}")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ablation.png", dpi=170)
    log(f"saved {out_dir}/ablation.csv, ablation_summary.txt, ablation.png")


if __name__ == "__main__":
    main()
