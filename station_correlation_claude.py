import pandas as pd
import matplotlib
import numpy as np
from scipy import stats
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── Load all stations (AE 440-870) ───────────────────────────────────────────
def load_csj():
    df = pd.read_csv(
        'PR AEROSOLS DATA/OneDrive_2_02-12-2025/AERONET_20040101_20251231_Cape_San_Juan/20040101_20251231_Cape_San_Juan.tot_lev20',
        skiprows=6
    )
    df["datetime"] = pd.to_datetime(df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"], format="%d:%m:%Y %H:%M:%S")
    for c in ["AOD_440nm-AOD", "AOD_870nm-AOD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["AOD_440nm-AOD","AOD_870nm-AOD"]] = df[["AOD_440nm-AOD","AOD_870nm-AOD"]].mask(
        df[["AOD_440nm-AOD","AOD_870nm-AOD"]].isin([-999.0, -9999.0]))
    df = df.set_index("datetime")
    mask = df["AOD_440nm-AOD"].notna() & df["AOD_870nm-AOD"].notna() & (df["AOD_440nm-AOD"] > 0) & (df["AOD_870nm-AOD"] > 0)
    df["AE"] = np.nan
    df.loc[mask, "AE"] = -np.log(df.loc[mask,"AOD_440nm-AOD"] / df.loc[mask,"AOD_870nm-AOD"]) / np.log(440/870)
    return df["AE"].resample("D").median().dropna()

def load_api(path):
    df = pd.read_csv(path, skiprows=5)
    df["datetime"] = pd.to_datetime(df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"], format="%d:%m:%Y %H:%M:%S")
    for c in ["AOD_440nm", "AOD_870nm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[["AOD_440nm","AOD_870nm"]] = df[["AOD_440nm","AOD_870nm"]].mask(df[["AOD_440nm","AOD_870nm"]].isin([-999.0,-9999.0]))
    df = df.set_index("datetime")
    mask = df["AOD_440nm"].notna() & df["AOD_870nm"].notna() & (df["AOD_440nm"] > 0) & (df["AOD_870nm"] > 0)
    df["AE"] = np.nan
    df.loc[mask, "AE"] = -np.log(df.loc[mask,"AOD_440nm"] / df.loc[mask,"AOD_870nm"]) / np.log(440/870)
    return df["AE"].resample("D").median().dropna()

csj = load_csj().rename("CSJ")
ng  = load_api('NEON_GUAN_all_years_lev20_daily.txt').rename("NG")
lp  = load_api('La_Parguera_all_years_lev20_daily.txt').rename("LP")

pairs = [
    (ng,  csj, "NEON Guánica",  "Cape San Juan", "#d94801", "#08519c"),
    (lp,  csj, "La Parguera",   "Cape San Juan", "#238b45", "#08519c"),
]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.35})

for ax, (s1, s2, lab1, lab2, c1, c2) in zip(axes, pairs):
    merged = pd.concat([s1, s2], axis=1).dropna()
    x_all, y_all = merged.iloc[:, 0].values, merged.iloc[:, 1].values

    keep = (x_all <= 2.0) & (y_all <= 2.0)
    x_all, y_all = x_all[keep], y_all[keep]

    n_all = len(x_all)
    slope, intercept, r, p, _ = stats.linregress(x_all, y_all)
    r2 = r**2
    rho, _ = stats.spearmanr(x_all, y_all)

    lim = (-0.1, 2.0)
    ax.scatter(x_all, y_all, s=8, alpha=0.25, color="#555555", linewidths=0, zorder=2)
    xl = np.array(lim)
    ax.plot(xl, slope * xl + intercept, color="#e31a1c", linewidth=1.5, zorder=3)


    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(lab1 + "\nAE 440–870 nm", fontsize=10)
    ax.set_ylabel(lab2 + "\nAE 440–870 nm", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.4)
    ax.set_facecolor("#f8f9fb")
    ax.text(0.05, 0.93,
            f"n = {n_all}\nPearson r = {r:.2f}  R² = {r2:.2f}\nSpearman ρ = {rho:.2f}\ny = {slope:.2f}x + {intercept:.2f}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))

fig.suptitle("AERONET Station Pair Correlations — AE 440–870 nm (daily median)",
             fontsize=13, fontweight="bold", y=0.98)

plt.savefig("station_correlation.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved station_correlation.png")
plt.show()
