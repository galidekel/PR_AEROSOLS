import cdsapi
from pathlib import Path

outdir = Path("/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS/era5_dataset")
outdir.mkdir(parents=True, exist_ok=True)

outfile = outdir / "era5_test.nc"

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": "2023",
        "month": "01",
        "day": "01",
        "time": "12:00",
        "format": "netcdf",
    },
    str(outfile),
)
print("Saved to:", outfile)