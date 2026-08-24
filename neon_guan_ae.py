import pandas as pd
import matplotlib
import numpy as np
import itertools
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Read file ---
df = pd.read_csv(
    "NEON_GUAN_all_years_lev20_daily.txt",
    skiprows=5
)

df["datetime"] = pd.to_datetime(
    df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)

plot_cols = [
    "AOD_440nm",
    "AOD_500nm",
    "AOD_870nm",
]

for c in plot_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[plot_cols] = df[plot_cols].mask(df[plot_cols].isin([-999.0, -9999.0]))
df = df.set_index("datetime")

# --- Ångström Exponent ---
ae_cols = []
wl_map = {col: int(col.split("_")[1].replace("nm", "")) for col in plot_cols}

for col_a, col_b in itertools.combinations(plot_cols, 2):
    wl_a, wl_b = wl_map[col_a], wl_map[col_b]
    ae_name = f"AE_{wl_a}_{wl_b}"
    mask = (
        df[col_a].notna() & df[col_b].notna() &
        (df[col_a] > 0) & (df[col_b] > 0)
    )
    df[ae_name] = np.nan
    df.loc[mask, ae_name] = (
        -np.log(df.loc[mask, col_a] / df.loc[mask, col_b])
        / np.log(wl_a / wl_b)
    )
    ae_cols.append(ae_name)

# --- Style ---
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
})

AOD_SCATTER = "#9ecae1"
AOD_MEAN    = "#08519c"
AE_SCATTER  = "#fdae6b"
AE_MEAN     = "#d94801"

AOD_LABELS = {
    "AOD_440nm": "AOD  440 nm",
    "AOD_500nm": "AOD  500 nm",
    "AOD_870nm": "AOD  870 nm",
}
AE_LABELS = {
    "AE_440_500": "AE (440–500 nm)",
    "AE_440_870": "AE (440–870 nm)",
    "AE_500_870": "AE (500–870 nm)",
}

n_aod   = len(plot_cols)
n_ae    = len(ae_cols)
n_total = n_aod + n_ae

fig, axes = plt.subplots(
    n_total, 1,
    figsize=(16, 2.8 * n_total),
    sharex=True,
    gridspec_kw={"hspace": 0.4},
)

for ax in axes[1:n_aod]:
    ax.sharey(axes[0])
for ax in axes[n_aod + 1:]:
    ax.sharey(axes[n_aod])

def daily_mean_centered(y):
    valid = y.dropna()
    grouped = valid.groupby(valid.index.normalize())
    vals  = grouped.mean().values
    times = grouped.apply(lambda x: x.index.mean()).values
    t_out, v_out = [times[0]], [vals[0]]
    for i in range(1, len(times)):
        if pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1]) > pd.Timedelta("1.5D"):
            t_out.append(pd.NaT)
            v_out.append(np.nan)
        t_out.append(times[i])
        v_out.append(vals[i])
    return pd.array(t_out, dtype="datetime64[ns]"), np.array(v_out, dtype=float)

# --- AOD panels ---
for i, col in enumerate(plot_cols):
    ax = axes[i]
    y = df[col]
    good_mask = y.notna()

    ax.scatter(df.index[good_mask], y[good_mask],
               s=10, alpha=0.45, color=AOD_SCATTER, linewidths=0, zorder=2)

    t_center, v_mean = daily_mean_centered(y)
    ax.plot(t_center, v_mean,
            linewidth=1.6, color=AOD_MEAN,
            marker="o", markersize=2.5, markeredgewidth=0,
            label="Daily mean", zorder=3)

    ax.set_ylabel(AOD_LABELS.get(col, col), fontsize=11, labelpad=8)
    ax.set_ylim(0, 1.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.grid(axis="y", linestyle=":",  linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_facecolor("#f8f9fb")

    if i == 0:
        ax.scatter([], [], s=40, color=AOD_SCATTER, alpha=0.7, label="Daily obs.")
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc",
                  loc="upper left", handletextpad=0.4)

# --- AE panels ---
for j, ae_col in enumerate(ae_cols):
    ax = axes[n_aod + j]
    y = df[ae_col]
    good_mask = y.notna()

    ax.scatter(df.index[good_mask], y[good_mask],
               s=10, alpha=0.45, color=AE_SCATTER, linewidths=0, zorder=2)

    t_center, v_mean = daily_mean_centered(y)
    ax.plot(t_center, v_mean,
            linewidth=1.6, color=AE_MEAN,
            marker="o", markersize=2.5, markeredgewidth=0,
            label="Daily mean", zorder=3)

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylabel(AE_LABELS.get(ae_col, ae_col), fontsize=11, labelpad=8)
    ax.set_ylim(-0.5, 3.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.grid(axis="y", linestyle=":",  linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_facecolor("#f8f9fb")

    if j == 0:
        ax.scatter([], [], s=40, color=AE_SCATTER, alpha=0.7, label="Daily obs.")
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc",
                  loc="upper left", handletextpad=0.4)

# --- x-axis ---
_locator   = mdates.AutoDateLocator(minticks=4, maxticks=9)
_formatter = mdates.ConciseDateFormatter(_locator)
axes[-1].xaxis.set_major_locator(_locator)
axes[-1].xaxis.set_major_formatter(_formatter)
axes[-1].tick_params(axis="x", which="major", labelsize=10)
axes[-1].set_xlabel("Date", fontsize=12, labelpad=6)

# --- AE y rescale on zoom ---
def _on_xlim_changed(event_ax):
    x0, x1 = event_ax.get_xlim()
    t0 = pd.Timestamp(mdates.num2date(x0)).tz_convert(None)
    t1 = pd.Timestamp(mdates.num2date(x1)).tz_convert(None)
    ae_vis = pd.concat([
        df.loc[(df.index >= t0) & (df.index <= t1), col].dropna()
        for col in ae_cols
    ])
    if len(ae_vis) >= 2:
        lo, hi = ae_vis.min(), ae_vis.max()
        pad = max((hi - lo) * 0.15, 0.1)
        axes[n_aod].set_ylim(lo - pad, hi + pad)
        for ax in axes[n_aod:]:
            ax.yaxis.set_major_locator(plt.MaxNLocator(5, min_n_ticks=3))

axes[0].callbacks.connect("xlim_changed", _on_xlim_changed)

axes[n_aod - 1].spines["bottom"].set_linewidth(1.5)
axes[n_aod - 1].spines["bottom"].set_color("#555555")

fig.suptitle(
    "AERONET — NEON Guánica, Puerto Rico\nAerosol Optical Depth & Ångström Exponent  (2017–2025)",
    fontsize=13, fontweight="bold", y=0.98,
)

fig.savefig("neon_guan_aod_ae.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved neon_guan_aod_ae.png")
plt.show()
