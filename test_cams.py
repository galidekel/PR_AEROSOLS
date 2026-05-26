import os
os.environ["CDSAPI_URL"] = "https://ads.atmosphere.copernicus.eu/api"
# If you rely on ~/.cdsapirc for ADS, keep it. Otherwise export CDSAPI_KEY in shell.

import cdsapi
from pathlib import Path

outdir = Path("/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS/cams_dataset")
outdir.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

c.retrieve(
    "cams-global-reanalysis-eac4",
    {
        "date": "2023-01-01",
        "time": ["12:00"],  # list is safer
        "variable": ["total_aerosol_optical_depth_550nm"],  # <- safer naming
        "format": "netcdf",
    },
    str(outdir / "cams_eac4_aod550_test.nc"),
)

print("done")

