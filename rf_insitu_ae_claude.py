import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# ── Load in-situ (scattering only, no AAE) ───────────────────────────────────
dfi = pd.read_excel('PR AEROSOLS DATA/OneDrive_2_02-12-2025/Aerosol optical properties_CSJ (1).xlsx')
dfi = dfi.iloc[1:].copy()
dfi = dfi.rename(columns={
    'Unnamed: 0': 'date',
    'Unnamed: 1': 'BsB_PM10',
    'PM_10':      'BsG_PM10',
    'Unnamed: 3': 'BsR_PM10',
    'Unnamed: 7': 'BsB_PM1',
    'PM_1':       'BsG_PM1',
    'Unnamed: 9': 'BsR_PM1',
})
dfi['date'] = pd.to_datetime(dfi['date'], errors='coerce')
dfi = dfi.dropna(subset=['date']).set_index('date')

feature_cols = ['BsB_PM10', 'BsG_PM10', 'BsR_PM10', 'SAE_PM10',
                'BsB_PM1',  'BsG_PM1',  'BsR_PM1',  'SAE_PM1']
for c in feature_cols:
    dfi[c] = pd.to_numeric(dfi[c], errors='coerce')
dfi[feature_cols] = dfi[feature_cols].replace([-999, -9999], pd.NA)
dfi_feat = dfi[feature_cols].dropna()

# ── Load CSJ AE 440-870 ───────────────────────────────────────────────────────
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

ae_median = df_csj["AE"].resample("D").median().dropna().rename("AE_median")
ae_mean   = df_csj["AE"].resample("D").mean().dropna().rename("AE_mean")
ae_min    = df_csj["AE"].resample("D").min().dropna().rename("AE_min")

targets = [
    (ae_median, "AE daily median"),
    (ae_mean,   "AE daily mean"),
    (ae_min,    "AE daily min"),
]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})
COLORS = ["#6a3d9a", "#1a9850", "#e31a1c"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"wspace": 0.35})

for ax, (ae_series, label), color in zip(axes, targets, COLORS):
    merged = pd.concat([dfi_feat, ae_series], axis=1).dropna()
    X = merged[feature_cols].values
    y = merged[ae_series.name].values
    n = len(y)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(rf, X, y, cv=cv)

    r2  = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r, _ = stats.pearsonr(y, y_pred)

    lim = (-0.1, 3.0)
    ax.scatter(y, y_pred, s=8, alpha=0.3, color=color, linewidths=0, zorder=2)
    ax.plot(lim, lim, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"Observed  ({label})", fontsize=10)
    ax.set_ylabel("RF predicted", fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.4)
    ax.set_facecolor("#f8f9fb")
    ax.text(0.05, 0.95,
            f"n = {n}\nR² = {r2:.2f}\nRMSE = {rmse:.3f}\nPearson r = {r:.2f}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

    print(f"{label:20s}  n={n:4d}  R²={r2:.3f}  RMSE={rmse:.3f}  r={r:.3f}")

fig.suptitle(
    "Random Forest — In-situ scattering features (PM₁₀ + PM₁) → AERONET AE 440–870 nm\n"
    "5-fold CV (shuffled)",
    fontsize=12, fontweight="bold", y=1.02
)

plt.savefig("rf_insitu_ae.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved rf_insitu_ae.png")
plt.show()
