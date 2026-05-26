import pandas as pd
import matplotlib
import json
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

# For each day and each AOD column: does the day contain >=1 non-NaN value?
# --- AOD columns with highest number of valid days ---

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
for col, n_days in valid_days_sorted[:20]:  # top 20 (change as you like)
    print(f"{col:25s}  {n_days}")
n_vars = len(plot_cols)
fig, axes = plt.subplots(n_vars, 1, figsize=(12, 2.5 * n_vars), sharex=True,sharey=True)

if n_vars == 1:
    axes = [axes]

for ax, col in zip(axes, plot_cols):
    y = df[col]
    x = df.index  # <-- datetime index

    good_mask = y.notna()

    # scatter all good points vs datetime
    ax.scatter(x[good_mask], y[good_mask], s=4, alpha=0.4, label="Measurements")

    # daily mean vs datetime
    daily_mean = y.resample("D").mean()  # resample by calendar day
    ax.plot(daily_mean.index, daily_mean.values, linewidth=1.8, c="red", label="Daily mean")

    ax.set_ylabel(col, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

axes[-1].set_xlabel("Datetime", fontsize=10)
fig.suptitle("AERONET variables vs time", fontsize=14)

axes[0].legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.show()
