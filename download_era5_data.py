#!/usr/bin/env python3
import os
import calendar
from pathlib import Path
import cdsapi
import argparse

# -----------------------
# USER SETTINGS
# -----------------------
OUTDIR = Path("/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS/era5_dataset")
OUTDIR.mkdir(parents=True, exist_ok=True)

# [North, West, South, East]  (example: Africa→Atlantic→Caribbean)
AREA = [35, -90, -10, 30]

START_YEAR = 2004
END_YEAR   = 2024

# 3-hourly
TIMES_3H = [f"{h:02d}:00" for h in range(0, 24, 3)]  # 00,03,...,21

# Pressure levels we want
PLEVELS = ["900", "850", "700", "500", "250"]


# -----------------------
# ERA5 DATASETS
# -----------------------
DS_PRESSURE = "reanalysis-era5-pressure-levels"
DS_SINGLE   = "reanalysis-era5-single-levels"

# Pressure-level variables (steering + synoptic pattern + RH)
# - u/v at 700/850
# - geopotential at 700/850
# - relative humidity at 700/850
PL_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "relative_humidity",

    # --- add to match the paper ---
    "temperature",          # T
    "specific_humidity",    # Q (note: this is NOT relative humidity)
    "vertical_velocity",    # W (pressure vertical velocity / omega-type; ERA5 naming is vertical_velocity)
    "potential_vorticity",  # PV
]

# Single-level variables (surface pressure + precip + PBL height)
# - msl: mean sea level pressure
# - tp: total precipitation (accumulated)
# - cp: convective precipitation (accumulated)
# - lsp: large-scale precipitation (accumulated)
# - blh: boundary layer height
SL_VARS = [
    "mean_sea_level_pressure",
    "total_precipitation",
    "convective_precipitation",
    "large_scale_precipitation",
    "boundary_layer_height",

    # --- add to match the paper-style inputs ---
    "10m_u_component_of_wind",     # u10
    "10m_v_component_of_wind",     # v10
    "total_column_water_vapour",   # tcwv
]

# -----------------------
def days_in_month(year: int, month: int):
    nd = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, nd + 1)]

def download_era5_month(c: cdsapi.Client, year: int, month: int):
    yyyymm = f"{year}{month:02d}"
    day_list = days_in_month(year, month)

    # -------- Pressure levels --------
    out_pl = OUTDIR / f"era5_pl_250_500_700_850_900_u_v_z_rh_t_q_w_pv_3h_{yyyymm}.nc"
    if not out_pl.exists() or out_pl.stat().st_size == 0:
        req_pl = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": PL_VARS,
            "pressure_level": PLEVELS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": day_list,
            "time": TIMES_3H,
            "area": AREA,
        }
        print(f"[get ] {out_pl.name}",flush=True)
        c.retrieve(DS_PRESSURE, req_pl, str(out_pl))
    else:
        print(f"[skip] {out_pl.name}",flush=True)

    # -------- Single levels --------
    out_sl = OUTDIR / f"era5_sl_mslp_tp_cp_lsp_blh_u10_v10_tcwv_3h_{yyyymm}.nc"
    if not out_sl.exists() or out_sl.stat().st_size == 0:
        req_sl = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": SL_VARS,
            "year": str(year),
            "month": f"{month:02d}",
            "day": day_list,
            "time": TIMES_3H,
            "area": AREA,
        }
        print(f"[get ] {out_sl.name}",flush=True)
        c.retrieve(DS_SINGLE, req_sl, str(out_sl))
    else:
        print(f"[skip] {out_sl.name}",flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None,
                        help="Download only this year (e.g., 2008). If omitted, downloads START_YEAR..END_YEAR.")
    parser.add_argument("--month", type=int, default=None,
                        help="Optional: download only this month (1-12). Requires --year.")
    args = parser.parse_args()

    c = cdsapi.Client()

    years = [args.year] if args.year is not None else list(range(START_YEAR, END_YEAR + 1))
    for year in years:
        if args.month is not None:
            if args.year is None:
                raise ValueError("--month requires --year")
            months = [args.month]
        else:
            months = range(1, 13)

        for month in months:
            download_era5_month(c, year, month)

    print("Done.")

if __name__ == "__main__":
    main()


