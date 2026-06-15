import pandas as pd
import numpy as np

AERONET_PATH = "/Users/galidek/PyCharmProjects/PR_Aerosols/PR AEROSOLS DATA/OneDrive_2_02-12-2025/AERONET_20040101_20251231_Cape_San_Juan/20040101_20251231_Cape_San_Juan.tot_lev20"

df = pd.read_csv(AERONET_PATH, skiprows=6)
df["datetime"] = pd.to_datetime(
    df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)
df = df.set_index("datetime")
df["AOD_500nm-AOD"] = pd.to_numeric(df["AOD_500nm-AOD"], errors="coerce")
df["AOD_500nm-AOD"] = df["AOD_500nm-AOD"].mask(df["AOD_500nm-AOD"].isin([-999.0, -9999.0]))
valid = df["AOD_500nm-AOD"][df["AOD_500nm-AOD"] > 0]
daily = valid.resample("D").median().dropna()

print(f"AOD_500nm median:     {daily.mean():.4f}")
print(f"AOD_500nm std:        {daily.std():.4f}")
print(f"AOD_500nm min/max:    {daily.min():.4f} / {daily.max():.4f}")
print(f"Baseline MSE:         {daily.var():.4f}")
print(f"Baseline RMSE:        {daily.std():.4f}")
print(f"Model val RMSE:       {0.0084**0.5:.4f}")
print(f"R²:                   {1 - 0.0084/daily.var():.4f}")
