import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


ZIP_PATH  = Path("era5_sl_mslp_tp_cp_lsp_blh_u10_v10_tcwv_3h_202412.nc")
CAMS_PATH = Path("cams_eac4_total_dust_aod550_3h_202412.nc")


# -------------------------------
# ERA5 loader
# -------------------------------
def open_era5_zip(zip_path: Path) -> xr.Dataset:
    extract_dir = zip_path.with_suffix("")
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    ds_inst = xr.open_dataset(extract_dir / "data_stream-oper_stepType-instant.nc")
    ds_acc  = xr.open_dataset(extract_dir / "data_stream-oper_stepType-accum.nc")

    ds = xr.merge([ds_inst, ds_acc], compat="override")

    ds = ds.assign(
        msl_hpa=ds["msl"] / 100.0,
        tp_mm=ds["tp"] * 1000.0,
        cp_mm=ds["cp"] * 1000.0,
        lsp_mm=ds["lsp"] * 1000.0,
    )

    return ds


# -------------------------------
# Downsample (optional)
# -------------------------------
def downsample(ds, step_lat=2, step_lon=2):
    return ds.isel(
        latitude=slice(None, None, step_lat),
        longitude=slice(None, None, step_lon),
    )


# -------------------------------
# 10-panel plot
# -------------------------------
def plot_10_panel(ds_era5, step_lat=2, step_lon=2):

    # ERA5 variables
    vars_era5 = ["msl_hpa", "u10", "v10", "blh",
                 "tcwv", "tp_mm", "cp_mm", "lsp_mm"]

    d = ds_era5[vars_era5]
    d = downsample(d, step_lat, step_lon)

    # CAMS
    ds_cams = xr.open_dataset(CAMS_PATH)
    cams = ds_cams[["aod550", "duaod550"]]

    # Align grids + time
    cams = cams.interp(
        latitude=d.latitude,
        longitude=d.longitude,
        valid_time=d.valid_time
    )

    # Choose time index
    t = 0

    data_list = [
        d.msl_hpa.isel(valid_time=t),
        d.u10.isel(valid_time=t),
        d.v10.isel(valid_time=t),
        d.blh.isel(valid_time=t),
        d.tcwv.isel(valid_time=t),
        d.tp_mm.isel(valid_time=t),
        d.cp_mm.isel(valid_time=t),
        d.lsp_mm.isel(valid_time=t),
        cams.aod550.isel(valid_time=t),
        cams.duaod550.isel(valid_time=t),
    ]

    titles = [
        "MSLP (hPa)", "U10 (m/s)", "V10 (m/s)",
        "BLH (m)", "TCWV (kg/m²)", "TP (mm)",
        "CP (mm)", "LSP (mm)", "AOD550",
        "Dust AOD"
    ]

    # -------------------------------
    # FIGURE
    # -------------------------------
    fig = plt.figure(figsize=(14, 16))

    for i, (data, title) in enumerate(zip(data_list, titles)):

        ax = plt.subplot(4, 3, i + 1, projection=ccrs.PlateCarree())

        ax.set_extent([-90, 30, -10, 35])  # Africa → Puerto Rico

        # Map features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="white")

        # Use data coordinates directly (FIXED)
        lats = data.latitude.values
        lons = data.longitude.values

        im = ax.pcolormesh(
            lons, lats, data.values,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            shading="auto"   # ← fixes your error
        )

        plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7)

        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.show()


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    ds = open_era5_zip(ZIP_PATH)
    plot_10_panel(ds)