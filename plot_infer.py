import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np

# --no-show skips the blocking plt.show() window at the end (still saves
# the PNG either way) — pass it when running non-interactively / no display.
show_plot = "--no-show" not in sys.argv
args = [a for a in sys.argv[1:] if a != "--no-show"]
in_csv = Path(args[0]) if args else Path("run2_infer.csv")
df = pd.read_csv(in_csv, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

pred = df["pred"].values
target = df["target"].values

# Metrics (computed on actual samples, not the gap-filled series below)
mae = np.mean(np.abs(pred - target))
rmse = np.sqrt(np.mean((pred - target) ** 2))
corr = np.corrcoef(pred, target)[0, 1]
ss_res = np.sum((target - pred) ** 2)
ss_tot = np.sum((target - target.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

# Reindex onto a full daily calendar so missing days become NaN gaps
# instead of being bridged by a straight interpolated line.
df_full = df.set_index("date").asfreq("D")
dates_full = df_full.index
pred_full = df_full["pred"].values
target_full = df_full["target"].values
resid_full = target_full - pred_full

has_cams = "cams_dust_aod" in df.columns and "cams_total_aod" in df.columns
if has_cams:
    cams_dust_full = df_full["cams_dust_aod"].values
    cams_total_full = df_full["cams_total_aod"].values
else:
    print("[plot_infer] no cams_dust_aod/cams_total_aod columns in CSV — "
          "re-run infer.py to add per-sample CAMS input averages. Skipping that overlay.")

# --- Style (matches main_with_ae.py) ---
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

GT_SCATTER   = "#c6dbef"   # light blue
GT_LINE      = "#6baed6"   # light blue
PRED_SCATTER = "#fed9a6"   # light orange
PRED_LINE    = "#fd8d3c"   # light orange
DUST_SCATTER  = "#c7e9c0"  # light green
DUST_LINE     = "#238b45"  # dark green
TOTAL_SCATTER = "#dadaeb"  # light purple
TOTAL_LINE    = "#6a51a3"  # dark purple
RESID_SCATTER = "#fdbe85"  # light orange (residual)
RESID_LINE    = "#e6550d"  # dark orange (residual)


def plot_points(ax, dates, values, scatter_color, line_color, label,
                linewidth=2.2, markersize=3.5, scatter_size=18, alpha=0.95):
    good = ~np.isnan(values)
    ax.scatter(dates[good], values[good], s=scatter_size, alpha=alpha * 0.55,
               color=scatter_color, linewidths=0, zorder=2)
    ax.plot(dates, values, linewidth=linewidth, color=line_color,
            marker="o", markersize=markersize, markeredgewidth=0,
            alpha=alpha, label=label, zorder=3)


n_rows = 5 if has_cams else 4
height_ratios = [2, 1.3, 1.3, 1, 4.8] if has_cams else [2, 1.3, 1, 4.3]
fig = plt.figure(figsize=(18, 26 if has_cams else 24), constrained_layout=True)
gs = fig.add_gridspec(n_rows, 3, height_ratios=height_ratios)
row = {"ts": 0,
       "cams": 1 if has_cams else None,
       "lag": 2 if has_cams else 1,
       "resid": 3 if has_cams else 2,
       "bottom": 4 if has_cams else 3}
fig.suptitle(
    f"AOD 500 Prediction vs Ground Truth\nMAE={mae:.4f}  RMSE={rmse:.4f}  r={corr:.3f}  R²={r2:.3f}",
    fontsize=13,
)

# --- Time series ---
ax = fig.add_subplot(gs[row["ts"], :])
plot_points(ax, dates_full, target_full, GT_SCATTER, GT_LINE, "Ground Truth")
plot_points(ax, dates_full, pred_full, PRED_SCATTER, PRED_LINE, "Predicted", alpha=0.75)
ax.set_ylabel("AOD")
ax.set_title("Predicted vs Ground Truth AOD Over Time", fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax.set_facecolor("#f8f9fb")

# --- CAMS input alone (dedicated panel, own y-scale) ---
if has_cams:
    ax_cams = fig.add_subplot(gs[row["cams"], :], sharex=ax)
    plot_points(ax_cams, dates_full, cams_dust_full, DUST_SCATTER, DUST_LINE,
                "CAMS dust AOD (input, avg)")
    plot_points(ax_cams, dates_full, cams_total_full, TOTAL_SCATTER, TOTAL_LINE,
                "CAMS total AOD (input, avg)")
    plot_points(ax_cams, dates_full, target_full, GT_SCATTER, GT_LINE,
                "Ground Truth", alpha=0.6)
    ax_cams.set_ylabel("AOD")
    ax_cams.set_title("CAMS Dust-Source Input (~T−7)", fontsize=11)
    plt.setp(ax_cams.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_cams.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax_cams.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax_cams.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
    ax_cams.set_facecolor("#f8f9fb")

# --- Lag correlation: does Predicted lead/lag Ground Truth? (+ CAMS input if available) ---
def lagged_corr(target_s, driver_s, max_lag=14):
    """corr(target[t], driver[t - lag]) for lag in [-max_lag, max_lag].
    Positive lag => driver value from `lag` days earlier is compared to target today
    (i.e. driver leads target by `lag` days when correlation peaks at positive lag)."""
    lags = range(-max_lag, max_lag + 1)
    corrs = [target_s.corr(driver_s.shift(lag)) for lag in lags]
    return list(lags), corrs

tgt_s = pd.Series(target_full, index=dates_full)
pred_s = pd.Series(pred_full, index=dates_full)

lags, corr_pred = lagged_corr(tgt_s, pred_s)
best_pred_lag = lags[int(np.nanargmax(corr_pred))]
print(f"[lag corr] target vs Predicted: peak r={max(corr_pred):.3f} at lag={best_pred_lag}d "
      f"({'predicted leads' if best_pred_lag > 0 else 'predicted lags' if best_pred_lag < 0 else 'in sync'})")

ax_lag = fig.add_subplot(gs[row["lag"], :])
ax_lag.axhline(0, color="black", linewidth=0.8)
ax_lag.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")
ax_lag.plot(lags, corr_pred, color=PRED_LINE, linewidth=2.2, marker="o",
            markersize=4, label=f"Predicted vs GT (peak r={max(corr_pred):.2f} @ {best_pred_lag}d)")

if has_cams:
    dust_s = pd.Series(cams_dust_full, index=dates_full)
    total_s = pd.Series(cams_total_full, index=dates_full)

    _, corr_dust = lagged_corr(tgt_s, dust_s)
    _, corr_total = lagged_corr(tgt_s, total_s)

    best_dust_lag = lags[int(np.nanargmax(corr_dust))]
    best_total_lag = lags[int(np.nanargmax(corr_total))]
    print(f"[lag corr] target vs CAMS dust AOD:  peak r={max(corr_dust):.3f} at lag={best_dust_lag}d")
    print(f"[lag corr] target vs CAMS total AOD: peak r={max(corr_total):.3f} at lag={best_total_lag}d")

    ax_lag.plot(lags, corr_dust, color=DUST_LINE, linewidth=2.0, marker="o",
                markersize=3.5, label=f"vs dust AOD (peak r={max(corr_dust):.2f} @ {best_dust_lag}d)")
    ax_lag.plot(lags, corr_total, color=TOTAL_LINE, linewidth=2.0, marker="o",
                markersize=3.5, label=f"vs total AOD (peak r={max(corr_total):.2f} @ {best_total_lag}d)")

ax_lag.set_xlabel("Lag (days)  —  positive = driver leads target")
ax_lag.set_ylabel("Correlation\nwith target AOD")
ax_lag.set_title("Lag Correlation with Target AOD", fontsize=11)
ax_lag.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")
ax_lag.grid(True, alpha=0.3)

# --- Residuals over time ---
ax_r = fig.add_subplot(gs[row["resid"], :], sharex=ax)
ax_r.axhline(0, color="black", linewidth=0.8)
plot_points(ax_r, dates_full, resid_full, RESID_SCATTER, RESID_LINE, "Residual (GT − Pred)")
roll = pd.Series(resid_full, index=dates_full).rolling(30, min_periods=10).mean()
ax_r.plot(dates_full, roll.values, color="black", linewidth=1.5, label="30-day rolling mean")
ax_r.set_ylabel("Residual\n(GT − Pred)")
ax_r.set_title("Residual Over Time", fontsize=11)
ax_r.set_xlabel("Date")
plt.setp(ax_r.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax_r.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
ax_r.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
ax_r.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4, zorder=0)
ax_r.set_facecolor("#f8f9fb")

# --- Scatter + error histograms: nested grid so the group sits centered
# as a block, instead of each axes independently centering within its own
# full-width share (which left an off-balance gap between them).
bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs[row["bottom"], :], width_ratios=[1, 1, 1]
)

# --- Scatter ---
ax2 = fig.add_subplot(bottom_gs[0, 0])
vmax = max(pred.max(), target.max()) * 1.05
ax2.scatter(target, pred, alpha=0.45, s=28, color="steelblue", edgecolors="none")
ax2.plot([0, vmax], [0, vmax], "k--", linewidth=1, label="1:1")
slope, intercept = np.polyfit(target, pred, 1)
x_fit = np.array([0, vmax])
ax2.plot(x_fit, slope * x_fit + intercept, color="tomato", linewidth=1.5,
          label=f"Fit (slope={slope:.2f}, b={intercept:.3f})")
ax2.set_xlabel("Ground Truth AOD")
ax2.set_ylabel("Predicted AOD")
ax2.set_title("Predicted vs Ground Truth", fontsize=11)
ax2.set_xlim(0, vmax)
ax2.set_ylim(0, vmax)
ax2.set_aspect("equal", anchor="C")
ax2.legend(loc="upper right", fontsize=7, framealpha=0.9, edgecolor="#cccccc")
ax2.grid(True, alpha=0.3)

# --- Residual distribution ---
ax3 = fig.add_subplot(bottom_gs[0, 1])
ax3.set_anchor("C")
ax3.hist(target - pred, bins=30, color="darkorange", alpha=0.75, edgecolor="white")
ax3.axvline(0, color="black", linewidth=1)
ax3.axvline((target - pred).mean(), color="black", linewidth=1.3, linestyle="--",
            label=f"mean={((target - pred).mean()):.4f}")
ax3.set_xlabel("Residual (GT − Pred)")
ax3.set_ylabel("Count")
ax3.set_title("Residual Distribution", fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Log-ratio error distribution (scale-relative, symmetric, no near-zero blowup) ---
ax4 = fig.add_subplot(bottom_gs[0, 2])
ax4.set_anchor("C")
std_resid = (target - pred) / target.std()
ax4.hist(std_resid, bins=30, color=PRED_LINE, alpha=0.75, edgecolor="white")
ax4.axvline(0, color="black", linewidth=1)
ax4.axvline(std_resid.mean(), color="black", linewidth=1.3, linestyle="--",
            label=f"mean={std_resid.mean():.3f}")
print(f"[std resid] mean((GT-Pred)/sigma_GT)^2 = {np.mean(std_resid ** 2):.3f}  "
      f"(compare to 1-R²={1 - r2:.3f})")
ax4.set_xlabel("Standardized Residual\n(GT − Pred) / σ_GT")
ax4.set_ylabel("Count")
ax4.set_title("Standardized Residual Distribution", fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3)

out = f"{in_csv.stem}_pred_vs_gt.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
if show_plot:
    plt.show()
