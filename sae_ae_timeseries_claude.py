import pandas as pd
import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load in-situ ─────────────────────────────────────────────────────────────
dfi = pd.read_excel('PR AEROSOLS DATA/OneDrive_2_02-12-2025/Aerosol optical properties_CSJ (1).xlsx')
dfi = dfi.iloc[1:].copy()
dfi = dfi.rename(columns={'Unnamed: 0': 'date'})
dfi['date'] = pd.to_datetime(dfi['date'], errors='coerce')
dfi = dfi.dropna(subset=['date']).set_index('date')

for c in ['SAE_PM10', 'SAE_PM1']:
    dfi[c] = pd.to_numeric(dfi[c], errors='coerce')
dfi[['SAE_PM10','SAE_PM1']] = dfi[['SAE_PM10','SAE_PM1']].replace([-999, -9999], pd.NA)

# ── Load AERONET AE 440-870 ───────────────────────────────────────────────────
df_csj = pd.read_csv(
    'PR AEROSOLS DATA/OneDrive_2_02-12-2025/AERONET_20040101_20251231_Cape_San_Juan/20040101_20251231_Cape_San_Juan.tot_lev20',
    skiprows=6
)
df_csj["datetime"] = pd.to_datetime(
    df_csj["Date(dd:mm:yyyy)"] + " " + df_csj["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)
for c in ["AOD_440nm-AOD", "AOD_870nm-AOD"]:
    df_csj[c] = pd.to_numeric(df_csj[c], errors="coerce")
df_csj[["AOD_440nm-AOD","AOD_870nm-AOD"]] = df_csj[["AOD_440nm-AOD","AOD_870nm-AOD"]].mask(
    df_csj[["AOD_440nm-AOD","AOD_870nm-AOD"]].isin([-999.0, -9999.0])
)
df_csj = df_csj.set_index("datetime")
mask = (df_csj["AOD_440nm-AOD"].notna() & df_csj["AOD_870nm-AOD"].notna()
        & (df_csj["AOD_440nm-AOD"] > 0) & (df_csj["AOD_870nm-AOD"] > 0))
df_csj["AE_440_870"] = np.nan
df_csj.loc[mask, "AE_440_870"] = (
    -np.log(df_csj.loc[mask, "AOD_440nm-AOD"] / df_csj.loc[mask, "AOD_870nm-AOD"])
    / np.log(440 / 870)
)

# daily median for AERONET
ae_daily = df_csj["AE_440_870"].resample("D").mean().dropna()

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

C_PM10 = "#2171b5"
C_PM1  = "#41ab5d"
C_AE   = "#d94801"

panels = [
    (dfi["SAE_PM10"], C_PM10, "SAE  PM₁₀"),
    (ae_daily,        C_AE,   "AE 440–870 nm  (AERONET daily mean)"),
]

fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True, sharey=True,
                          gridspec_kw={"hspace": 0.35})

def with_gap_breaks(series, max_gap_days=1.5):
    s = series.dropna().sort_index()
    if len(s) == 0:
        return pd.Series(dtype=float)
    idx, vals = list(s.index), list(s.values)
    t_out, v_out = [], []
    for i, (t, v) in enumerate(zip(idx, vals)):
        if i > 0 and (t - idx[i-1]).days > max_gap_days:
            t_out.append(idx[i-1] + (t - idx[i-1]) / 2)
            v_out.append(np.nan)
        t_out.append(t)
        v_out.append(v)
    return pd.Series(v_out, index=pd.DatetimeIndex(t_out))

for ax, (series, color, label) in zip(axes, panels):
    line_s = with_gap_breaks(series)
    ax.plot(line_s.index, line_s.values,
            color=color, linewidth=0.8, alpha=0.6, zorder=2)
    ax.scatter(series.dropna().index, series.dropna().values,
               s=10, alpha=0.7, color=color, linewidths=0, zorder=3)
    ax.set_ylabel(label, fontsize=10.5, labelpad=8)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.grid(axis="y", linestyle=":",  linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_facecolor("#f8f9fb")
    ax.set_ylim(-0.2, 3.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))

_locator   = mdates.AutoDateLocator(minticks=4, maxticks=10)
_formatter = mdates.ConciseDateFormatter(_locator)
axes[-1].xaxis.set_major_locator(_locator)
axes[-1].xaxis.set_major_formatter(_formatter)
axes[-1].tick_params(axis="x", labelsize=10)
axes[-1].set_xlabel("Date", fontsize=12, labelpad=6)

fig.suptitle(
    "Cape San Juan, Puerto Rico\n"
    "In-situ SAE PM₁₀ and AERONET AE 440–870 nm",
    fontsize=13, fontweight="bold", y=0.99,
)

plt.savefig("sae_ae_timeseries.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved sae_ae_timeseries.png")
plt.show()
