"""
Plot the two attention gates' weights, extracted by analyze_attention.py.

Gate 2 (fusion): bar chart of mean attention weight per source
    (dust_source + each route/meteo channel) -> which inputs matter most.
Gate 1 (per-channel space-time): for the top-K channels by fusion weight,
    a spatial heatmap (where in the domain) + a temporal profile
    (when in the T_in-day window) of average attention.

Usage:
    python plot_attention.py <attention_dir> [--top_k 6] [--no-show]

<attention_dir> must contain fusion_weights.csv and route_attention.npz,
as produced by `python analyze_attention.py --out_dir <attention_dir>`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PR_LON, PR_LAT = -65.6199, 18.3845   # Cape San Juan AERONET station


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attention_dir", help="Dir with fusion_weights.csv + route_attention.npz")
    parser.add_argument("--top_k", type=int, default=6, help="Number of top channels to detail (Gate 1)")
    parser.add_argument("--no-show", action="store_true", help="Skip the blocking plt.show() window")
    args = parser.parse_args()

    d = Path(args.attention_dir)
    fusion_df = pd.read_csv(d / "fusion_weights.csv", parse_dates=["date"])
    npz = np.load(d / "route_attention.npz", allow_pickle=True)

    channel_names = list(npz["channel_names"])
    Tp, Hp, Wp = int(npz["Tp"]), int(npz["Hp"]), int(npz["Wp"])
    N, W, S, E = [float(v) for v in npz["area_route"]]
    t_step_hours = float(npz["t_step_hours"])
    n_samples = int(npz["n_samples"])
    token_cols = [c for c in fusion_df.columns if c not in ("date", "pred", "target")]

    # ── Gate 2: fusion importance bar chart ────────────────────────────────
    mean_w = fusion_df[token_cols].mean().sort_values(ascending=False)
    std_w  = fusion_df[token_cols].std().reindex(mean_w.index)

    fig1, ax1 = plt.subplots(figsize=(max(8, 0.32 * len(token_cols)), 5))
    colors = ["#d94801" if name == "dust_source" else "#2171b5" for name in mean_w.index]
    ax1.bar(range(len(mean_w)), mean_w.values, yerr=std_w.values,
            color=colors, capsize=2, error_kw={"linewidth": 0.8, "alpha": 0.6})
    ax1.set_xticks(range(len(mean_w)))
    ax1.set_xticklabels(mean_w.index, rotation=90, fontsize=7)
    ax1.set_ylabel("Mean attention weight (Gate 2 — fusion)")
    ax1.set_title(f"Fusion gate: which inputs does the PR token attend to?\n"
                   f"(n={len(fusion_df)} val samples; orange = dust source, blue = route/meteo channel)",
                   fontsize=15)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out1 = d / "fusion_importance.png"
    plt.savefig(out1, dpi=180, bbox_inches="tight")
    print(f"Saved {out1}")

    # ── Gate 1: top-K channels — spatial map + temporal profile ───────────
    top_channels = [name for name in mean_w.index if name in channel_names][: args.top_k]
    print(f"Top {len(top_channels)} channels by fusion weight: {top_channels}")

    lat_centers = np.linspace(S, N, Hp + 1)
    lon_centers = np.linspace(W, E, Wp + 1)

    n = len(top_channels)
    fig2 = plt.figure(figsize=(4.6 * n, 7.5))
    for i, ch in enumerate(top_channels):
        spatial  = npz[f"spatial__{ch}"]     # [Hp, Wp]
        temporal = npz[f"temporal__{ch}"]    # [Tp]

        # Spatial heatmap — land/ocean fill drawn FIRST, attention overlay is
        # semi-transparent so the basemap (and coastlines on top) stay visible
        # underneath, instead of a fully opaque grid hiding the geography.
        # Row 1 = all maps side by side; row 2 = all temporal profiles below them.
        ax_map = fig2.add_subplot(2, n, i + 1, projection=ccrs.PlateCarree())
        ax_map.set_extent([W, E, S, N], crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.OCEAN, facecolor="#cfe8f3", zorder=0)
        ax_map.add_feature(cfeature.LAND, facecolor="#f2e5c4", zorder=1)
        mesh = ax_map.pcolormesh(lon_centers, lat_centers, spatial, cmap="inferno",
                                  transform=ccrs.PlateCarree(), zorder=2, shading="flat",
                                  alpha=0.72)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="white", zorder=3)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="white", alpha=0.6, zorder=3)
        ax_map.scatter(PR_LON, PR_LAT, s=45, color="#00e676", edgecolors="black",
                        linewidths=0.7, marker="*", transform=ccrs.PlateCarree(), zorder=5,
                        label="Cape San Juan")
        fig2.colorbar(mesh, ax=ax_map, fraction=0.03, pad=0.02, label="attn")
        ax_map.set_title(f"{ch}", fontsize=13, fontweight="bold")
        gl = ax_map.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.3)
        if i == 0:
            ax_map.legend(loc="lower left", fontsize=6.5, framealpha=0.85)

        # Temporal profile — same column, second row
        ax_t = fig2.add_subplot(2, n, n + i + 1)
        hours_before = (Tp - 1 - np.arange(Tp)) * t_step_hours
        ax_t.bar(range(Tp), temporal, color="#2171b5", width=0.85)
        ax_t.set_xticks(range(0, Tp, max(1, Tp // 6)))
        ax_t.set_xticklabels(
            [f"-{int(h)}h" for h in hours_before[::max(1, Tp // 6)]], fontsize=6.5, rotation=60
        )
        ax_t.set_xlabel("time before target day", fontsize=7.5)
        if i == 0:
            ax_t.set_ylabel("attn (temporal)", fontsize=8)
        ax_t.set_title(f"{ch} — temporal", fontsize=13, fontweight="bold")
        ax_t.grid(axis="y", alpha=0.3)

    fig2.suptitle(f"Gate 1 — per-channel space-time attention (top {len(top_channels)} channels, "
                   f"averaged over {n_samples} val samples)", fontsize=17, fontweight="bold", y=1.03)
    plt.tight_layout()
    out2 = d / "route_attention_top_channels.png"
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    print(f"Saved {out2}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
