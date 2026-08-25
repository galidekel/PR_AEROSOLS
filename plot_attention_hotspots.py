"""
Answer "which patch, at which time" for one Gate-1 channel, using the joint
[Tp, Hp, Wp] attention pattern saved by analyze_attention.py (route_attention.npz
must have been produced by a version that saves joint__<channel> -- re-run
analyze_attention.py if you only have an older file with just spatial_/temporal_).

1) Prints a ranked table of the top-N individual (time, lat, lon) attention cells.
2) Plots a filmstrip: one small map per hottest individual time-step (not
   time-averaged), so you can see the actual spatial pattern at that moment.

Usage:
    python plot_attention_hotspots.py <attention_dir> --channel pv_700
    python plot_attention_hotspots.py <attention_dir> --channel pv_700 --top_n 15 --n_frames 6
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

PR_LON, PR_LAT = -65.6199, 18.3845   # Cape San Juan AERONET station


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attention_dir")
    parser.add_argument("--channel", required=True, help="Route channel name, e.g. pv_700")
    parser.add_argument("--top_n", type=int, default=10, help="Rows in the ranked (time,lat,lon) table")
    parser.add_argument("--n_frames", type=int, default=6, help="Number of hottest time-steps to map")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    d = Path(args.attention_dir)
    npz = np.load(d / "route_attention.npz", allow_pickle=True)

    key = f"joint__{args.channel}"
    if key not in npz:
        raise SystemExit(
            f"'{key}' not found in {d/'route_attention.npz'} -- this file was made before "
            f"analyze_attention.py saved the joint tensor. Re-run analyze_attention.py and "
            f"re-copy route_attention.npz, then try again."
        )
    joint = npz[key]   # [Tp, Hp, Wp]
    Tp, Hp, Wp = joint.shape
    N, W, S, E = [float(v) for v in npz["area_route"]]
    t_step_hours = float(npz["t_step_hours"])
    n_samples = int(npz["n_samples"])

    lat_edges = np.linspace(S, N, Hp + 1)
    lon_edges = np.linspace(W, E, Wp + 1)
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    hours_before = (Tp - 1 - np.arange(Tp)) * t_step_hours

    # ── Ranked table of individual (time, lat, lon) cells ──────────────────
    flat_idx = np.argsort(joint.ravel())[::-1][: args.top_n]
    t_idx, h_idx, w_idx = np.unravel_index(flat_idx, joint.shape)

    print(f"\nTop {args.top_n} (time, patch) cells for '{args.channel}' "
          f"(averaged over {n_samples} val samples):")
    print(f"{'rank':>4}  {'time before target':>19}  {'lat':>7}  {'lon':>8}  {'attn':>9}  {'x mean':>7}")
    mean_cell = joint.mean()
    for rank, (t, h, w) in enumerate(zip(t_idx, h_idx, w_idx), 1):
        val = joint[t, h, w]
        print(f"{rank:>4}  {f'-{int(hours_before[t])}h':>19}  {lat_c[h]:>7.2f}  {lon_c[w]:>8.2f}  "
              f"{val:>9.5f}  {val / mean_cell:>6.1f}x")

    # ── Filmstrip: hottest individual time-steps, each its own map ─────────
    per_t_total = joint.sum(axis=(1, 2))
    top_t = np.argsort(per_t_total)[::-1][: args.n_frames]
    top_t = np.sort(top_t)   # chronological order left-to-right

    vmax = joint.max()
    fig, axes = plt.subplots(
        1, len(top_t), figsize=(3.3 * len(top_t), 3.6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if len(top_t) == 1:
        axes = [axes]

    for ax, t in zip(axes, top_t):
        ax.set_extent([W, E, S, N], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#cfe8f3", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f2e5c4", zorder=1)
        mesh = ax.pcolormesh(lon_edges, lat_edges, joint[t], cmap="inferno",
                              vmin=0, vmax=vmax, transform=ccrs.PlateCarree(),
                              zorder=2, shading="flat", alpha=0.75)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="white", zorder=3)
        ax.scatter(PR_LON, PR_LAT, s=40, color="#00e676", edgecolors="black",
                   linewidths=0.6, marker="*", transform=ccrs.PlateCarree(), zorder=5)
        ax.set_title(f"-{int(hours_before[t])}h", fontsize=11, fontweight="bold")

    fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02, label="attn (this time-step)")
    fig.suptitle(f"'{args.channel}' — spatial pattern AT its {len(top_t)} hottest individual time-steps "
                 f"(not time-averaged)", fontsize=13, fontweight="bold", y=1.05)

    out = d / f"{args.channel}_hotspots.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
