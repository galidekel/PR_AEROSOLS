import pandas as pd
import matplotlib
import numpy as np
import itertools
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load Cape San Juan ───────────────────────────────────────────────────────
df_csj = pd.read_csv(
    'PR AEROSOLS DATA/OneDrive_2_02-12-2025/AERONET_20040101_20251231_Cape_San_Juan/20040101_20251231_Cape_San_Juan.tot_lev20',
    skiprows=6
)
df_csj["datetime"] = pd.to_datetime(
    df_csj["Date(dd:mm:yyyy)"] + " " + df_csj["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)
csj_cols = ["AOD_440nm-AOD", "AOD_500nm-AOD", "AOD_870nm-AOD"]
for c in csj_cols:
    df_csj[c] = pd.to_numeric(df_csj[c], errors="coerce")
df_csj[csj_cols] = df_csj[csj_cols].mask(df_csj[csj_cols].isin([-999.0, -9999.0]))
df_csj = df_csj.set_index("datetime")[csj_cols]
df_csj.columns = ["AOD_440", "AOD_500", "AOD_870"]

# ── Load NEON GUAN ───────────────────────────────────────────────────────────
df_ng = pd.read_csv(
    'NEON_GUAN_all_years_lev20_daily.txt',
    skiprows=5
)
df_ng["datetime"] = pd.to_datetime(
    df_ng["Date(dd:mm:yyyy)"] + " " + df_ng["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)
ng_cols = ["AOD_440nm", "AOD_500nm", "AOD_870nm"]
for c in ng_cols:
    df_ng[c] = pd.to_numeric(df_ng[c], errors="coerce")
df_ng[ng_cols] = df_ng[ng_cols].mask(df_ng[ng_cols].isin([-999.0, -9999.0]))
df_ng = df_ng.set_index("datetime")[ng_cols]
df_ng.columns = ["AOD_440", "AOD_500", "AOD_870"]

# ── Load La Parguera ──────────────────────────────────────────────────────────
df_lp = pd.read_csv(
    'La_Parguera_all_years_lev20_daily.txt',
    skiprows=5
)
df_lp["datetime"] = pd.to_datetime(
    df_lp["Date(dd:mm:yyyy)"] + " " + df_lp["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)
lp_cols = ["AOD_440nm", "AOD_500nm", "AOD_870nm"]
for c in lp_cols:
    df_lp[c] = pd.to_numeric(df_lp[c], errors="coerce")
df_lp[lp_cols] = df_lp[lp_cols].mask(df_lp[lp_cols].isin([-999.0, -9999.0]))
df_lp = df_lp.set_index("datetime")[lp_cols]
df_lp.columns = ["AOD_440", "AOD_500", "AOD_870"]

# ── Compute AE for both ──────────────────────────────────────────────────────
def add_ae(df):
    pairs = [(440, 500, "AOD_440", "AOD_500"), (440, 870, "AOD_440", "AOD_870"), (500, 870, "AOD_500", "AOD_870")]
    for wl_a, wl_b, col_a, col_b in pairs:
        ae_name = f"AE_{wl_a}_{wl_b}"
        mask = df[col_a].notna() & df[col_b].notna() & (df[col_a] > 0) & (df[col_b] > 0)
        df[ae_name] = np.nan
        df.loc[mask, ae_name] = (
            -np.log(df.loc[mask, col_a] / df.loc[mask, col_b])
            / np.log(wl_a / wl_b)
        )
    return df

df_csj = add_ae(df_csj)
df_ng  = add_ae(df_ng)
df_lp  = add_ae(df_lp)

aod_cols = []
ae_cols  = ["AE_440_870", "AE_500_870"]

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

CSJ_COLOR  = "#08519c"
NG_COLOR   = "#d94801"
LP_COLOR   = "#238b45"
CSJ_MEAN   = "#08306b"
NG_MEAN    = "#a63603"
LP_MEAN    = "#00441b"

AOD_LABELS = {"AOD_440": "AOD  440 nm", "AOD_500": "AOD  500 nm", "AOD_870": "AOD  870 nm"}
AE_LABELS  = {"AE_440_500": "AE (440–500 nm)", "AE_440_870": "AE (440–870 nm)", "AE_500_870": "AE (500–870 nm)"}

n_total = len(ae_cols)
fig, axes = plt.subplots(n_total, 1, figsize=(16, 5),
                          sharex=True, sharey=True, gridspec_kw={"hspace": 0.4})

def daily_median_line(y, max_gap_days=1.5):
    valid = y.dropna()
    if len(valid) == 0:
        return pd.Series(dtype=float)
    daily = valid.resample("D").median().dropna()
    idx = list(daily.index)
    vals = list(daily.values)
    t_out, v_out = [], []
    for i, (t, v) in enumerate(zip(idx, vals)):
        if i > 0 and (t - idx[i-1]).days > max_gap_days:
            t_out.append(idx[i-1] + (t - idx[i-1]) / 2)
            v_out.append(np.nan)
        t_out.append(t)
        v_out.append(v)
    return pd.Series(v_out, index=pd.DatetimeIndex(t_out))

# ── AE panels ────────────────────────────────────────────────────────────────
for j, col in enumerate(ae_cols):
    ax = axes[j]
    # CSJ scatter dots
    good = df_csj[col].notna()
    ax.scatter(df_csj.index[good], df_csj[col][good],
               s=8, alpha=0.08, color=CSJ_COLOR, linewidths=0, zorder=2)
    for df, color, label in [
        (df_csj, CSJ_COLOR, "Cape San Juan"),
        (df_ng,  NG_COLOR,  "NEON Guánica"),
        (df_lp,  LP_COLOR,  "La Parguera"),
    ]:
        s = daily_median_line(df[col])
        ax.plot(s.index, s.values, linewidth=1.2, color=color,
                label=f"Daily median — {label}" if j == 0 else None, zorder=3)

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylabel(AE_LABELS[col], fontsize=11, labelpad=8)
    ax.set_ylim(-0.5, 3.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_facecolor("#f8f9fb")
    if j == 0:
        ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#cccccc",
                  loc="upper left", handletextpad=0.4)

# ── x-axis ───────────────────────────────────────────────────────────────────
_locator   = mdates.AutoDateLocator(minticks=4, maxticks=9)
_formatter = mdates.ConciseDateFormatter(_locator)
axes[-1].xaxis.set_major_locator(_locator)
axes[-1].xaxis.set_major_formatter(_formatter)
axes[-1].tick_params(axis="x", labelsize=10)
axes[-1].set_xlabel("Date", fontsize=12, labelpad=6)


fig.suptitle(
    "AERONET — Cape San Juan (blue),  NEON Guánica (orange)  &  La Parguera (green), Puerto Rico\n"
    "Aerosol Optical Depth & Ångström Exponent",
    fontsize=13, fontweight="bold", y=0.98,
)

fig.savefig("combined_stations_aod_ae.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved combined_stations_aod_ae.png")
plt.show()
