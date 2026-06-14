import pandas as pd
import numpy as np

AERONET_PATH = "/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS_DATA/20040101_20251231_Cape_San_Juan.tot_lev20"

df = pd.read_csv(AERONET_PATH, skiprows=6)
df["datetime"] = pd.to_datetime(
    df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
    format="%d:%m:%Y %H:%M:%S"
)
df = df.set_index("datetime")
df["AOD_440nm-AOD"] = pd.to_numeric(df["AOD_440nm-AOD"], errors="coerce")
df["AOD_870nm-AOD"] = pd.to_numeric(df["AOD_870nm-AOD"], errors="coerce")
mask = (df["AOD_440nm-AOD"] > 0) & (df["AOD_870nm-AOD"] > 0)
ae = -np.log(df.loc[mask, "AOD_440nm-AOD"] / df.loc[mask, "AOD_870nm-AOD"]) / np.log(440 / 870)
daily = ae.resample("D").mean().dropna()

print(f"AE mean:              {daily.mean():.4f}")
print(f"AE std:               {daily.std():.4f}")
print(f"AE min/max:           {daily.min():.4f} / {daily.max():.4f}")
print(f"Baseline MSE (mean):  {daily.var():.4f}")
print(f"Baseline RMSE:        {daily.std():.4f}")
