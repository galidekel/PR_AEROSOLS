import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np

ds_cams = xr.open_dataset("cams_eac4_total_dust_aod550_3h_202412.nc")
dust = ds_cams["aod550"].sel(longitude=slice(-17, 30), latitude=slice(35, -10))

# pick 8 snapshots spread across December, including the peak event
T_IDXS = [0, 24, 48, 72, 96, 120, 134, 168, 192, 216, 240, 247]

proj  = ccrs.PlateCarree()
ncols = 4
nrows = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 11),
                          subplot_kw={"projection": proj})
axes = axes.flatten()

vmax = 0.6
cmap = "YlOrRd"
C_OCEAN = "#cfe8f3"
C_LAND  = "#f2e5c4"

for ax, t_idx in zip(axes, T_IDXS):
    field = dust.isel(valid_time=t_idx)
    t_str = str(ds_cams.valid_time.values[t_idx])[:13].replace("T", "\n")

    ax.set_extent([-17, 30, -10, 35], crs=proj)
    ax.add_feature(cfeature.OCEAN,     facecolor=C_OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor=C_LAND,  zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,     zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, alpha=0.5, zorder=3)
    gl = ax.gridlines(linewidth=0.3, color="gray", alpha=0.4, linestyle=":")
    gl.top_labels = gl.right_labels = False

    im = ax.pcolormesh(field.longitude, field.latitude, field.values,
                       cmap=cmap, vmin=0, vmax=vmax,
                       transform=proj, zorder=2)
    ax.set_title(t_str, fontsize=8.5, fontweight="bold", pad=3)

# shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                    fraction=0.015, pad=0.02, shrink=0.85)
cbar.set_label("Total AOD 550 nm", fontsize=10)

fig.suptitle("CAMS EAC4 — Total AOD over Africa  (December 2024)",
             fontsize=13, fontweight="bold", y=1.01)

plt.savefig("aod_fields_africa.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved aod_fields_africa.png")
plt.show()
