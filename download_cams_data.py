#!/usr/bin/env python3
"""
Download CAMS EAC4 (ADS) for:
  - total AOD 550 nm
  - dust AOD 550 nm

Saves one NetCDF per month, 3-hourly (00,03,...,21) for 2004–2024.

Run on the remote server.

Prereqs (remote shell):
  export CDSAPI_URL="https://ads.atmosphere.copernicus.eu/api"
  export CDSAPI_KEY="YOUR_ADS_TOKEN"
  pip install cdsapi
"""
import os
os.environ["CDSAPI_URL"] = "https://ads.atmosphere.copernicus.eu/api"
import calendar
from pathlib import Path
import cdsapi

# -----------------------
# User settings
# -----------------------
OUTDIR = Path("/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS/cams_dataset")
OUTDIR.mkdir(parents=True, exist_ok=True)

# [North, West, South, East] (edit as you like; this is Africa→Atlantic→Caribbean)
AREA = [35, -90, -10, 30]

START_YEAR = 2004
END_YEAR = 2024

TIMES_3H = [f"{h:02d}:00" for h in range(0, 24, 3)]  # 00:00, 03:00, ..., 21:00

VARS = [
    "total_aerosol_optical_depth_550nm",
    "dust_aerosol_optical_depth_550nm",
]

DATASET = "cams-global-reanalysis-eac4"
# -----------------------


def days_in_month(year: int, month: int) -> list[str]:
    ndays = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, ndays + 1)]


def download_month(c: cdsapi.Client, year: int, month: int) -> Path:
    yyyymm = f"{year}{month:02d}"
    outfile = OUTDIR / f"cams_eac4_total_dust_aod550_3h_{yyyymm}.nc"

    # Skip if already downloaded
    if outfile.exists() and outfile.stat().st_size > 0:
        print(f"[skip] {outfile.name}")
        return outfile
    import calendar

    def month_date_range(year: int, month: int) -> str:
        last = calendar.monthrange(year, month)[1]
        return f"{year}-{month:02d}-01/{year}-{month:02d}-{last:02d}"

    # ...

    req = {
        "variable": VARS,
        "date": month_date_range(year, month),  # <-- ADS style
        "time": TIMES_3H,
        "area": AREA,
        "format": "netcdf",
    }

    print(f"[get ] {outfile.name}")
    c.retrieve(DATASET, req, str(outfile))
    return outfile


def main():
    c = cdsapi.Client()  # uses CDSAPI_URL/CDSAPI_KEY env vars
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            download_month(c, year, month)
    print("Done.")


if __name__ == "__main__":
    main()
