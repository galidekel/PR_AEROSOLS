"""
CAMS dust-source input average over every year in the dataset -- ONE figure, styled like the
"CAMS Dust-Source Input (avg)" panel of plot_infer.py, but without the ground truth.

The quantity matches infer.py's 'cams_dust_aod' / 'cams_total_aod': the plain spatial mean of CAMS
AOD550 (dust = duaod550, total = aod550) over area_africa, on the model's 6-hourly time steps
(00/06/12/18 UTC), averaged per day. One difference by design: infer.py dates each value with the
TARGET date (the value was measured ~6 days earlier); here every value is dated with the day it
was actually measured.

Two steps in one script:
  1. COMPUTE (needs the CAMS NetCDF files -- run on the HPC) -> cams_dust_daily_mean.csv
  2. PLOT (needs only the CSV -- run anywhere)

Usage:
    # on the HPC (writes the CSV and the PNG):
    python plot_cams_dust_mean.py --config config_hpc.yaml
    # on the Mac, after copying cams_dust_daily_mean.csv down (plot only):
    python plot_cams_dust_mean.py --csv cams_dust_daily_mean.csv
Options: --all_hours (use every 3-hourly step), --weighted (cos-lat weighted mean), --no-show.

To use EXACTLY the data that enters the model (recommended), read the memmap the model trains on:
    python plot_cams_dust_mean.py --memmap_dir /home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS_DATA/memmap
This also prints the shape of the model's dust grid and the fraction of NaN / zero cells.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Same colors/style as plot_infer.py
DUST_SCATTER, DUST_LINE = "#c7e9c0", "#238b45"
TOTAL_SCATTER, TOTAL_LINE = "#dadaeb", "#6a51a3"


def compute_daily_mean(cfg, weighted, all_hours):
    from train_main import load_all_files, crop_area          # heavy imports only when computing
    ds = load_all_files(Path(cfg["cams_dir"]), cfg["pattern_cams"], chunks={"time": cfg["chunk_time"]})
    ds = crop_area(ds, cfg["area_africa"])
    if not all_hours:                                          # the model's time grid (t_step_hours)
        step = cfg["t_step_hours"]
        ds = ds.sel(time=ds.time.dt.hour % step == 0)
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    out = {}
    for name, var in (("cams_dust_aod", "duaod550"), ("cams_total_aod", "aod550")):
        da = ds[var]
        if weighted:
            reg = da.weighted(np.cos(np.deg2rad(ds[lat]))).mean(dim=[lat, lon])
        else:
            reg = da.mean(dim=[lat, lon])
        out[name] = reg.resample(time="1D").mean().compute().to_series()
        # Regional MAXIMUM (the peak grid cell anywhere in the box) at each step, then averaged per day.
        out[name.replace("_aod", "_max")] = (da.max(dim=[lat, lon]).resample(time="1D").mean()
                                             .compute().to_series())
    df = pd.DataFrame(out).dropna()          # resample() leaves empty days where there is no CAMS file
    df.index.name = "date"
    return df.reset_index()


def compute_from_memmap(memmap_dir):
    """Read EXACTLY what enters the model: dust.dat = [T, 2, H_d, W_d] float32 (channel 0 = total AOD550,
    channel 1 = dust AOD550), already cropped to area_africa and on the 6-hourly grid, before z-scoring.
    NaNs are set to 0 first, as PRLazyDataset does before the model sees the data."""
    import json
    d = Path(memmap_dir)
    meta = json.loads((d / "meta.json").read_text())
    T, C, H, W = meta["dust_shape"]
    mm = np.memmap(d / "dust.dat", dtype="float32", mode="r", shape=(T, C, H, W))
    times = pd.to_datetime(meta["times"])
    print(f"[memmap] dust.dat shape T={T} C={C} H={H} W={W}   {times.min()} → {times.max()}")

    mean = np.empty((T, C)); mx = np.empty((T, C)); n_nan = 0; n_zero = 0
    for s in range(0, T, 2000):
        x = np.asarray(mm[s:s + 2000])
        n_nan += int(np.isnan(x).sum())
        x = np.nan_to_num(x, nan=0.0)
        n_zero += int((x == 0).sum())
        mean[s:s + 2000] = x.mean(axis=(2, 3))
        mx[s:s + 2000] = x.max(axis=(2, 3))
    tot = T * C * H * W
    print(f"[memmap] NaN cells: {n_nan} ({100 * n_nan / tot:.2f}%)   exactly-zero cells after NaN->0: "
          f"{n_zero} ({100 * n_zero / tot:.2f}%)")
    df = pd.DataFrame({"cams_total_aod": mean[:, 0], "cams_dust_aod": mean[:, 1],
                       "cams_total_max": mx[:, 0], "cams_dust_max": mx[:, 1]}, index=times)
    df = df.resample("1D").mean().dropna()
    df.index.name = "date"
    return df.reset_index()


def plot_points(ax, dates, values, scatter_color, line_color, label,
                linewidth=1.3, markersize=2.2, scatter_size=7, alpha=0.95):
    """Same look as plot_infer.plot_points (scatter + line with markers), smaller marks for a long record."""
    good = ~np.isnan(values)
    ax.scatter(dates[good], values[good], s=scatter_size, alpha=alpha * 0.55,
               color=scatter_color, linewidths=0, zorder=2)
    ax.plot(dates, values, linewidth=linewidth, color=line_color, marker="o", markersize=markersize,
            markeredgewidth=0, alpha=alpha, label=label, zorder=3)


def _one_figure(full, dust_col, total_col, ylabel, title, out_png):
    """One single-panel figure in plot_infer.py's CAMS-panel style (dust green, total purple)."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates = full.index
    fig, ax = plt.subplots(figsize=(18, 4.8))
    plot_points(ax, dates, full[dust_col].values, DUST_SCATTER, DUST_LINE, f"CAMS dust AOD ({ylabel})")
    plot_points(ax, dates, full[total_col].values, TOTAL_SCATTER, TOTAL_LINE, f"CAMS total AOD ({ylabel})")
    ax.set_ylabel("AOD")
    ax.set_title(f"{title} — {dates.min().date()} to {dates.max().date()}", fontsize=11)
    n_years = dates.max().year - dates.min().year + 1
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1 if n_years <= 10 else 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_facecolor("#f8f9fb")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}   (dust {full[dust_col].mean():.3f} ± {full[dust_col].std():.3f};"
          f" total {full[total_col].mean():.3f} ± {full[total_col].std():.3f})")


def make_plot(df, out_png, show):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8})
    full = df.sort_values("date").set_index("date").asfreq("D")        # missing days -> NaN gaps (like plot_infer)

    # Figure 1: regional AVERAGE
    _one_figure(full, "cams_dust_aod", "cams_total_aod", "input, avg",
                "CAMS Dust-Source Input (avg over the Africa region)", out_png)

    # Figure 2: regional MAXIMUM (separate figure: its scale is much larger than the average's)
    if {"cams_dust_max", "cams_total_max"} <= set(full.columns):
        out_max = str(Path(out_png).with_name(Path(out_png).stem + "_max.png"))
        _one_figure(full, "cams_dust_max", "cams_total_max", "regional max",
                    "CAMS Dust-Source Input (maximum grid cell in the Africa region, daily mean of the 6-hourly max)",
                    out_max)
    else:
        print("No cams_*_max columns in this CSV (computed with the older version) -- "
              "re-run in compute mode on the HPC to get the max figure.")
    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_hpc.yaml", help="Config with cams_dir / area_africa (compute mode)")
    p.add_argument("--csv", default=None, help="Plot from an existing CSV instead of recomputing")
    p.add_argument("--memmap_dir", default=None,
                   help="Compute from the model's own input file (dust.dat + meta.json in this dir, e.g. the "
                        "memmap_dir in the config) instead of the raw CAMS NetCDFs")
    p.add_argument("--out_csv", default="cams_dust_daily_mean.csv")
    p.add_argument("--out_png", default=None)
    p.add_argument("--weighted", action="store_true", help="cos(latitude)-weighted regional mean")
    p.add_argument("--all_hours", action="store_true", help="Use every 3-hourly step, not just the model's 6-hourly grid")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    if a.csv:
        df = pd.read_csv(a.csv, parse_dates=["date"])
        out_png = a.out_png or str(Path(a.csv).with_suffix(".png"))
    else:
        if a.memmap_dir:
            df = compute_from_memmap(a.memmap_dir)
        else:
            from train_main import load_config
            df = compute_daily_mean(load_config(a.config), a.weighted, a.all_hours)
        df.to_csv(a.out_csv, index=False)
        print(f"Saved {a.out_csv}  ({len(df)} days, {df['date'].min().date()} → {df['date'].max().date()})")
        out_png = a.out_png or str(Path(a.out_csv).with_suffix(".png"))
    make_plot(df, out_png, show=not a.no_show)


if __name__ == "__main__":
    main()
