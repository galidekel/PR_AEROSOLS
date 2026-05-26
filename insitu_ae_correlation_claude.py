import pandas as pd
import matplotlib
import numpy as np
from scipy import stats
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── Load in-situ SAE ──────────────────────────────────────────────────────────
dfi = pd.read_excel('PR AEROSOLS DATA/OneDrive_2_02-12-2025/Aerosol optical properties_CSJ (1).xlsx')
dfi = dfi.iloc[1:].copy()
dfi = dfi.rename(columns={'Unnamed: 0': 'date'})
dfi['date'] = pd.to_datetime(dfi['date'], errors='coerce')
dfi = dfi.dropna(subset=['date']).set_index('date')
for c in ['SAE_PM10', 'SAE_PM1']:
    dfi[c] = pd.to_numeric(dfi[c], errors='coerce')
dfi[['SAE_PM10','SAE_PM1']] = dfi[['SAE_PM10','SAE_PM1']].replace([-999, -9999], pd.NA)
sae_pm10 = dfi['SAE_PM10'].dropna()
sae_pm1  = dfi['SAE_PM1'].dropna()

# ── Load CSJ AE 440-870 (daily median) ───────────────────────────────────────
df_csj = pd.read_csv(
    'PR AEROSOLS DATA/OneDrive_2_02-12-2025/AERONET_20040101_20251231_Cape_San_Juan/20040101_20251231_Cape_San_Juan.tot_lev20',
    skiprows=6
)
df_csj["datetime"] = pd.to_datetime(
    df_csj["Date(dd:mm:yyyy)"] + " " + df_csj["Time(hh:mm:ss)"], format="%d:%m:%Y %H:%M:%S"
)
for c in ["AOD_440nm-AOD", "AOD_870nm-AOD"]:
    df_csj[c] = pd.to_numeric(df_csj[c], errors="coerce")
df_csj[["AOD_440nm-AOD","AOD_870nm-AOD"]] = df_csj[["AOD_440nm-AOD","AOD_870nm-AOD"]].mask(
    df_csj[["AOD_440nm-AOD","AOD_870nm-AOD"]].isin([-999.0, -9999.0]))
df_csj = df_csj.set_index("datetime")
mask = (df_csj["AOD_440nm-AOD"].notna() & df_csj["AOD_870nm-AOD"].notna()
        & (df_csj["AOD_440nm-AOD"] > 0) & (df_csj["AOD_870nm-AOD"] > 0))
df_csj["AE"] = np.nan
df_csj.loc[mask, "AE"] = (
    -np.log(df_csj.loc[mask,"AOD_440nm-AOD"] / df_csj.loc[mask,"AOD_870nm-AOD"])
    / np.log(440/870)
)
csj_ae = df_csj["AE"].resample("D").mean().dropna()

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

pairs = [
    (sae_pm10, csj_ae, "SAE  PM₁₀  (in-situ)", "Cape San Juan AE 440–870 nm", "#2171b5"),
]

fig, axes = plt.subplots(1, 1, figsize=(6, 5))
axes = [axes]

for ax, (sx, sy, labx, laby, color) in zip(axes, pairs):
    merged = pd.concat([sx.rename("x"), sy.rename("y")], axis=1).dropna()
    x, y = merged["x"].values, merged["y"].values
    keep = (x <= 2.0) & (y <= 2.0)
    x, y = x[keep], y[keep]

    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2 = r**2
    rho, _ = stats.spearmanr(x, y)

    lim = (-0.1, 2.0)
    ax.scatter(x, y, s=8, alpha=0.25, color=color, linewidths=0, zorder=2)
    xl = np.array(lim)
    ax.plot(xl, slope * xl + intercept, color="#e31a1c", linewidth=1.5, zorder=3)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(labx, fontsize=10)
    ax.set_ylabel(laby, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.4)
    ax.set_facecolor("#f8f9fb")
    ax.text(0.05, 0.93,
            f"n = {len(x)}\nPearson r = {r:.2f}  R² = {r2:.2f}\nSpearman ρ = {rho:.2f}\ny = {slope:.2f}x + {intercept:.2f}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

fig.suptitle("In-situ SAE vs AERONET AE 440–870 nm\n(Cape San Juan, daily mean)",
             fontsize=12, fontweight="bold", y=1.02)

plt.savefig("insitu_ae_correlation.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved insitu_ae_correlation.png")
plt.show()
