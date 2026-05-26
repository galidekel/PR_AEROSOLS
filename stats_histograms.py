import pandas as pd
import matplotlib
import numpy as np
import itertools
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load & prep ────────────────────────────────────────────────────────────
df = pd.read_csv(
    'PR AEROSOLS DATA/OneDrive_2_02-12-2025/AERONET_20040101_20251231_Cape_San_Juan/20040101_20251231_Cape_San_Juan.tot_lev20',
    skiprows=6
)
df["datetime"] = pd.to_datetime(
    df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)
plot_cols = ["AOD_440nm-AOD", "AOD_500nm-AOD", "AOD_870nm-AOD"]
for c in plot_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[plot_cols] = df[plot_cols].mask(df[plot_cols].isin([-999.0, -9999.0]))
df = df.set_index("datetime")

# ── Ångström Exponent ──────────────────────────────────────────────────────
ae_cols = []
wl_map = {col: int(col.split("_")[1].replace("nm-AOD", "")) for col in plot_cols}
for col_a, col_b in itertools.combinations(plot_cols, 2):
    wl_a, wl_b = wl_map[col_a], wl_map[col_b]
    ae_name = f"AE_{wl_a}_{wl_b}"
    mask = df[col_a].notna() & df[col_b].notna() & (df[col_a] > 0) & (df[col_b] > 0)
    df[ae_name] = np.nan
    df.loc[mask, ae_name] = (
        -np.log(df.loc[mask, col_a] / df.loc[mask, col_b])
        / np.log(wl_a / wl_b)
    )
    ae_cols.append(ae_name)

# ── Daily aggregations ─────────────────────────────────────────────────────
all_cols = plot_cols + ae_cols
daily_median = df[all_cols].resample("D").median()
daily_max    = df[all_cols].resample("D").max()
daily_min    = df[all_cols].resample("D").min()

# ── Labels ────────────────────────────────────────────────────────────────
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

ROWS = [
    (daily_median, plot_cols, AOD_LABELS, "#4292c6", "Daily Median  AOD"),
    (daily_max,    plot_cols, AOD_LABELS, "#cb181d", "Daily Max  AOD"),
    (daily_median, ae_cols,   AE_LABELS,  "#f16913", "Daily Median  AE"),
    (daily_min,    ae_cols,   AE_LABELS,  "#41ab5d", "Daily Min  AE"),
]

MEAN_COLOR = "#08306b"
MED_COLOR  = "#7f0000"

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

fig, axes = plt.subplots(
    4, 3,
    figsize=(14, 13),
    gridspec_kw={"hspace": 0.55, "wspace": 0.32},
)

# ── Draw each row ─────────────────────────────────────────────────────────
for row_idx, (daily, cols, labels, color, row_title) in enumerate(ROWS):
    # share y within this row
    for col_idx in range(1, 3):
        axes[row_idx, col_idx].sharey(axes[row_idx, 0])

    for col_idx, col in enumerate(cols):
        ax   = axes[row_idx, col_idx]
        vals = daily[col].dropna().values

        is_aod = col in plot_cols
        if is_aod:
            AOD_CAP = 1.0
            plot_vals = np.clip(vals, 0.0, AOD_CAP)
            bins = np.linspace(0.0, AOD_CAP, 52)
            n_above = (vals > AOD_CAP).sum()
        else:
            plot_vals = vals
            bins = np.linspace(vals.min(), vals.max(), 60)
            n_above = 0

        ax.hist(plot_vals, bins=bins, color=color, edgecolor="white",
                linewidth=0.5, alpha=0.85, zorder=2)

        mean = vals.mean()
        med  = np.median(vals)
        ax.axvline(mean, color=MEAN_COLOR, linewidth=1.4, linestyle="--",
                   label=f"Mean = {mean:.3f}", zorder=3)
        ax.axvline(med,  color=MED_COLOR,  linewidth=1.4, linestyle=":",
                   label=f"Median = {med:.3f}", zorder=3)

        ax.set_xlabel(labels[col], fontsize=9.5, labelpad=4)
        ax.set_ylabel("Days", fontsize=9.5, labelpad=4)
        ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
                  handlelength=1.3, handletextpad=0.3)
        ax.text(0.97, 0.95, f"n = {len(vals)} days",
                transform=ax.transAxes, fontsize=7.5, color="#555555",
                ha="right", va="top")
        ax.set_facecolor("#f8f9fb")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)

        if is_aod:
            # label last tick ">1" and report how many days landed there
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", ">1"])
            if n_above > 0:
                ax.text(0.98, 0.78, f"({n_above} days > 1)",
                        transform=ax.transAxes, fontsize=7, color="#cc0000",
                        ha="right", va="top")

    # Row label on the left of the first column
    axes[row_idx, 0].set_ylabel("Days", fontsize=9.5, labelpad=4)
    axes[row_idx, 0].annotate(
        row_title,
        xy=(-0.22, 0.5), xycoords="axes fraction",
        fontsize=10, fontweight="bold", color=color,
        ha="right", va="center", rotation=90,
    )

# ── Horizontal dividers between sections ──────────────────────────────────
for row_idx in [1]:   # single divider between AOD (rows 0-1) and AE (rows 2-3)
    y_fig = axes[row_idx, 0].get_position().y1 + 0.005
    line = plt.Line2D([0.05, 0.95], [y_fig, y_fig],
                      transform=fig.transFigure,
                      color="#aaaaaa", linewidth=0.8, linestyle="--")
    fig.add_artist(line)

fig.suptitle(
    "AERONET — Cape San Juan, Puerto Rico\n"
    "Histograms of Daily AOD & Ångström Exponent  (2004–2025)",
    fontsize=12, fontweight="bold", y=0.995,
)

plt.savefig("aeronet_hist_combined.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved aeronet_hist_combined.png")

# ══════════════════════════════════════════════════════════════════════════════
# IN-SITU figure — Aerosol optical properties_CSJ (1).xlsx
# Columns:
#   SAE_PM10   – Scattering Ångström Exponent, PM10
#   AAE_PM10.1 – Absorption Ångström Exponent, PM10
#   SAE_PM1    – Scattering Ångström Exponent, PM1
#   AAE_PM1.1  – Absorption Ångström Exponent, PM1
# ══════════════════════════════════════════════════════════════════════════════
INSITU_FILE = "PR AEROSOLS DATA/OneDrive_2_02-12-2025/Aerosol optical properties_CSJ (1).xlsx"
dfi = pd.read_excel(INSITU_FILE)
dfi = dfi.iloc[1:].copy()
dfi = dfi.rename(columns={"Unnamed: 0": "date"})
dfi["date"] = pd.to_datetime(dfi["date"], errors="coerce")

insitu_cols = {
    "SAE_PM10":   "SAE  PM₁₀  (scattering AE)",
    "AAE_PM10.1": "AAE  PM₁₀  (absorption AE)",
    "SAE_PM1":    "SAE  PM₁  (scattering AE)",
    "AAE_PM1.1":  "AAE  PM₁  (absorption AE)",
}
for c in insitu_cols:
    dfi[c] = pd.to_numeric(dfi[c], errors="coerce")
dfi[list(insitu_cols)] = dfi[list(insitu_cols)].replace([-999, -9999], pd.NA)

INSITU_COLORS = {
    "SAE_PM10":   "#2171b5",
    "AAE_PM10.1": "#6a3d9a",
    "SAE_PM1":    "#41ab5d",
    "AAE_PM1.1":  "#d94801",
}

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 7),
                            gridspec_kw={"hspace": 0.45, "wspace": 0.35})
axes2 = axes2.flatten()

for ax, (col, label) in zip(axes2, insitu_cols.items()):
    vals = dfi[col].dropna().values
    color = INSITU_COLORS[col]
    bins = np.linspace(vals.min(), vals.max(), 60)

    ax.hist(vals, bins=bins, color=color, edgecolor="white",
            linewidth=0.5, alpha=0.85, zorder=2)

    mean = vals.mean()
    med  = np.median(vals)
    ax.axvline(mean, color=MEAN_COLOR, linewidth=1.4, linestyle="--",
               label=f"Mean = {mean:.3f}", zorder=3)
    ax.axvline(med,  color=MED_COLOR,  linewidth=1.4, linestyle=":",
               label=f"Median = {med:.3f}", zorder=3)

    ax.set_xlabel(label, fontsize=10, labelpad=4)
    ax.set_ylabel("Days", fontsize=10, labelpad=4)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc",
              handlelength=1.3, handletextpad=0.3)
    ax.text(0.97, 0.95, f"n = {len(vals)} days",
            transform=ax.transAxes, fontsize=8, color="#555555",
            ha="right", va="top")
    ax.set_facecolor("#f8f9fb")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig2.suptitle(
    "In-situ — Cape San Juan, Puerto Rico\n"
    "Scattering & Absorption Ångström Exponent  (PM₁₀ and PM₁)",
    fontsize=12, fontweight="bold", y=0.995,
)

plt.savefig("insitu_hist.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved insitu_hist.png")
plt.show()
