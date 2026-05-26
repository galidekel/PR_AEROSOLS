import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np

ds_cams = xr.open_dataset("cams_eac4_total_dust_aod550_3h_202412.nc")
ds_pl   = xr.open_dataset("era5_pl_250_500_700_850_900_u_v_z_rh_t_q_w_pv_3h_202412.nc")

T_IDXS = [0, 24, 48, 72, 96, 120, 134, 168, 192, 216, 240, 247]

u700 = ds_pl["u"].sel(pressure_level=700)
v700 = ds_pl["v"].sel(pressure_level=700)
wspd = np.sqrt(u700**2 + v700**2)

lat = ds_pl.latitude.values
lon = ds_pl.longitude.values
SKIP = 10   # quiver subsampling

proj   = ccrs.PlateCarree()
ncols  = 4
nrows  = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 11),
                          subplot_kw={"projection": proj})
axes = axes.flatten()

vmin, vmax = 0, 20
cmap    = "Blues"
C_OCEAN = "#cfe8f3"
C_LAND  = "#f2e5c4"

for ax, t_idx in zip(axes, T_IDXS):
    spd   = wspd.isel(valid_time=t_idx).values
    u_t   = u700.isel(valid_time=t_idx).values
    v_t   = v700.isel(valid_time=t_idx).values
    t_str = str(ds_cams.valid_time.values[t_idx])[:13].replace("T", "\n")

    ax.set_extent([-90, 30, -10, 35], crs=proj)
    ax.add_feature(cfeature.OCEAN,     facecolor=C_OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor=C_LAND,  zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,     zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, alpha=0.5, zorder=3)
    gl = ax.gridlines(linewidth=0.3, color="gray", alpha=0.4, linestyle=":")
    gl.top_labels = gl.right_labels = False

    im = ax.pcolormesh(lon, lat, spd,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=proj, zorder=2)

    # wind vectors
    lats_s = lat[::SKIP]
    lons_s = lon[::SKIP]
    LON_G, LAT_G = np.meshgrid(lons_s, lats_s)
    ax.quiver(LON_G, LAT_G,
              u_t[::SKIP, ::SKIP], v_t[::SKIP, ::SKIP],
              transform=proj, zorder=4,
              scale=350, width=0.002, color="#08306b", alpha=0.6)

    # Cape San Juan
    ax.scatter(-65.62, 18.38, s=25, color="red", edgecolors="darkred",
               linewidths=0.5, transform=proj, zorder=5)

    ax.set_title(t_str, fontsize=8.5, fontweight="bold", pad=3)

cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                    fraction=0.015, pad=0.02, shrink=0.85)
cbar.set_label("Wind speed 700 hPa  (m/s)", fontsize=10)

fig.suptitle("ERA5 — Wind Speed & Direction at 700 hPa  (December 2024)",
             fontsize=13, fontweight="bold", y=1.01)

plt.savefig("era5_wind700_fields.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved era5_wind700_fields.png")
plt.show()
