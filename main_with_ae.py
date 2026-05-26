import pandas as pd
import matplotlib
import json
import numpy as np
import itertools
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# --- Read file ---
df = pd.read_csv(
    'PR AEROSOLS DATA/OneDrive_2_02-12-2025/AERONET_20040101_20251231_Cape_San_Juan/20040101_20251231_Cape_San_Juan.tot_lev20',
    skiprows=6
)
pd.set_option('display.max_columns', None)
print(df.columns)
# --- Create datetime BEFORE numeric conversion ---
df["datetime"] = pd.to_datetime(
    df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)
print(df.columns.tolist())
# --- Choose the columns we care about ---
cols = [
    "datetime",
    "Date(dd:mm:yyyy)",
    "Time(hh:mm:ss)",
    "Day_of_Year",
    "AOD_440nm-AOD",
    "AOD_500nm-AOD",
    "AOD_675nm-AOD",
    "AOD_870nm-AOD",
    "AOD_1020nm-AOD"

]
df = df[cols]

# --- Convert only numeric columns to numeric ---
numeric_cols = [
     "Day_of_Year",
    "AOD_440nm-AOD",
    "AOD_500nm-AOD",
    "AOD_675nm-AOD",
    "AOD_870nm-AOD",
    "AOD_1020nm-AOD"

]

for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Mark AERONET missing flags as NaN ---
df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols].isin([-999.0, -9999.0]))

# --- Optional: valid counts dict ---
valid_counts = df[numeric_cols].notna().sum()
valid_counts_dict = valid_counts[valid_counts > 0].to_dict()
with open('valid_counts.json', 'w') as f:
    json.dump(valid_counts_dict, f, indent=4)

# --- Use datetime as index ---
df = df.set_index("datetime").drop(columns=["Date(dd:mm:yyyy)", "Time(hh:mm:ss)"])

# -------- PLOTTING VS DATETIME --------
plot_cols = [
    "AOD_440nm-AOD",
    "AOD_500nm-AOD",
    "AOD_870nm-AOD",
]

# pick all AOD "*-AOD" columns automatically (safer than hardcoding)
aod_cols = [c for c in df.columns if c.startswith("AOD_") and c.endswith("-AOD")]

# count unique days with >=1 valid value per column
valid_days_per_col = {
    col: df.loc[df[col].notna()].index.normalize().nunique()
    for col in aod_cols
}

# sort descending
valid_days_sorted = sorted(valid_days_per_col.items(), key=lambda kv: kv[1], reverse=True)

print("\nAOD columns ranked by # of valid days (>=1 measurement/day):")
for col, n_days in valid_days_sorted[:20]:
    print(f"{col:25s}  {n_days}")

# --- Compute Angstrom Exponent for all wavelength pairs in plot_cols ---
ae_cols = []
wl_map = {col: int(col.split("_")[1].replace("nm-AOD", "")) for col in plot_cols}

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
    valid_days_ae = df.loc[df[ae_name].notna()].index.normalize().nunique()
    print(f"{ae_name:25s}  {valid_days_ae} valid days")
    ae_cols.append(ae_name)

# --- Style ---
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
})

# Palette
AOD_SCATTER  = "#9ecae1"   # light blue
AOD_MEAN     = "#08519c"   # dark blue
AE_SCATTER   = "#fdae6b"   # light orange
AE_MEAN      = "#d94801"   # dark orange

# Readable labels
AOD_LABELS = {
    "AOD_440nm-AOD": "AOD  440 nm",
    "AOD_500nm-AOD": "AOD  500 nm",
    "AOD_870nm-AOD": "AOD  870 nm",
}
AE_LABELS = {
    "AE_440_500": "AE (440–500 nm)",
    "AE_440_870": "AE (440–870 nm)",
    "AE_500_870": "AE (500–870 nm)",
}

# --- Layout ---
n_aod   = len(plot_cols)
n_ae    = len(ae_cols)
n_total = n_aod + n_ae

fig, axes = plt.subplots(
    n_total, 1,
    figsize=(16, 2.8 * n_total),
    sharex=True,
    gridspec_kw={"hspace": 0.4},
)

# Share y-axes within each group so limits are always identical
for ax in axes[1:n_aod]:
    ax.sharey(axes[0])
for ax in axes[n_aod + 1:]:
    ax.sharey(axes[n_aod])

def daily_mean_centered(y):
    """Returns (mean_timestamps, mean_values) with NaT/NaN breaks at gaps > 1 day."""
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
        ax.scatter([], [], s=40, color=AOD_SCATTER, alpha=0.7, label="Sub-daily obs.")
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
        ax.scatter([], [], s=40, color=AE_SCATTER, alpha=0.7, label="Sub-daily obs.")
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc",
                  loc="upper left", handletextpad=0.4)

# --- Shared x-axis formatting: auto-adapts ticks/labels at every zoom level ---
import matplotlib.dates as mdates
_locator   = mdates.AutoDateLocator(minticks=4, maxticks=9)
_formatter = mdates.ConciseDateFormatter(_locator)
axes[-1].xaxis.set_major_locator(_locator)
axes[-1].xaxis.set_major_formatter(_formatter)
axes[-1].tick_params(axis="x", which="major", labelsize=10)
axes[-1].set_xlabel("Date", fontsize=12, labelpad=6)

# --- AE y-axis rescales to visible data on zoom; AOD stays fixed at 0–1.5 ---
def _on_xlim_changed(event_ax):
    x0, x1 = event_ax.get_xlim()
    t0 = pd.Timestamp(mdates.num2date(x0)).tz_convert(None)
    t1 = pd.Timestamp(mdates.num2date(x1)).tz_convert(None)
    # combine visible data across all 3 AE columns for a shared range
    ae_vis = pd.concat([
        df.loc[(df.index >= t0) & (df.index <= t1), col].dropna()
        for col in ae_cols
    ])
    if len(ae_vis) >= 2:
        lo, hi = ae_vis.min(), ae_vis.max()
        pad = max((hi - lo) * 0.15, 0.1)
        axes[n_aod].set_ylim(lo - pad, hi + pad)   # propagates to all AE via sharey
        for ax in axes[n_aod:]:
            ax.yaxis.set_major_locator(plt.MaxNLocator(5, min_n_ticks=3))

axes[0].callbacks.connect("xlim_changed", _on_xlim_changed)

# --- Divider line between AOD and AE sections ---
axes[n_aod - 1].spines["bottom"].set_linewidth(1.5)
axes[n_aod - 1].spines["bottom"].set_color("#555555")

fig.suptitle(
    "AERONET — Cape San Juan, Puerto Rico\nAerosol Optical Depth & Ångström Exponent  (2004–2025)",
    fontsize=13, fontweight="bold", y=0.98,
)

fig.savefig("aeronet_aod_ae.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.show()
