import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np

# ── load data ───────────────────────────────────────────────────────────────
T_IDX = 134   # 2024-12-17 18:00 — peak dust event

ds_cams = xr.open_dataset("cams_eac4_total_dust_aod550_3h_202412.nc")
ds_pl   = xr.open_dataset("era5_pl_250_500_700_850_900_u_v_z_rh_t_q_w_pv_3h_202412.nc")
ds_sl   = xr.open_dataset("era5_sl_mslp_tp_cp_lsp_blh_u10_v10_tcwv_3h_202412.nc_unzipped/data_stream-oper_stepType-instant.nc")

t_label = str(ds_cams.valid_time.values[T_IDX])[:16].replace("T", " ")

# ── domains ─────────────────────────────────────────────────────────────────
# full route
rN, rW, rS, rE = 35, -90, -10, 30
# africa dust
aN, aW, aS, aE = 35, -17, -10, 30

# ── extract fields at T_IDX ──────────────────────────────────────────────────
dust   = ds_cams["duaod550"].isel(valid_time=T_IDX)
aod    = ds_cams["aod550"].isel(valid_time=T_IDX)
u700   = ds_pl["u"].isel(valid_time=T_IDX).sel(pressure_level=700)
v700   = ds_pl["v"].isel(valid_time=T_IDX).sel(pressure_level=700)
q850   = ds_pl["q"].isel(valid_time=T_IDX).sel(pressure_level=850) * 1000  # kg/kg → g/kg
msl    = ds_sl["msl"].isel(valid_time=T_IDX) / 100   # Pa → hPa

# crop CAMS to Africa
dust_af = dust.sel(longitude=slice(aW, aE), latitude=slice(aN, aS))
aod_af  = aod.sel(longitude=slice(aW, aE),  latitude=slice(aN, aS))

lat_pl = ds_pl.latitude.values
lon_pl = ds_pl.longitude.values
lat_sl = ds_sl.latitude.values
lon_sl = ds_sl.longitude.values

# coarsen ERA5 fields for cleaner wind vectors
SKIP = 8

# ── figure ───────────────────────────────────────────────────────────────────
proj = ccrs.PlateCarree()
fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                          subplot_kw={"projection": proj})

C_OCEAN = "#cfe8f3"
C_LAND  = "#f2e5c4"

def basemap(ax, extent):
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.OCEAN,     facecolor=C_OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND,      facecolor=C_LAND,  zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6,     zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, alpha=0.5, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.4, linestyle=":")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

PR_LON, PR_LAT = -65.62, 18.38

# ── Panel 1: CAMS Dust AOD over Africa ──────────────────────────────────────
ax = axes[0]
basemap(ax, [aW, aE, aS, aN])
im1 = ax.pcolormesh(dust_af.longitude, dust_af.latitude, dust_af.values,
                    cmap="YlOrRd", vmin=0, vmax=0.6,
                    transform=proj, zorder=2)
plt.colorbar(im1, ax=ax, orientation="horizontal", pad=0.04,
             label="Dust AOD 550 nm", shrink=0.85)
ax.set_title("CAMS — Dust AOD (Africa domain)", fontsize=10, fontweight="bold")
ax.text(0.02, 0.97, t_label, transform=ax.transAxes,
        fontsize=8, va="top", color="#333333",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

# ── Panel 2: ERA5 Wind at 700 hPa + wind speed ──────────────────────────────
ax = axes[1]
basemap(ax, [rW, rE, rS, rN])

wspd = np.sqrt(u700.values**2 + v700.values**2)
im2 = ax.pcolormesh(lon_pl, lat_pl, wspd,
                    cmap="Blues", vmin=0, vmax=20,
                    transform=proj, zorder=2)
plt.colorbar(im2, ax=ax, orientation="horizontal", pad=0.04,
             label="Wind speed (m/s)", shrink=0.85)

# wind vectors (subsampled)
lats_s = lat_pl[::SKIP]
lons_s = lon_pl[::SKIP]
u_s    = u700.values[::SKIP, ::SKIP]
v_s    = v700.values[::SKIP, ::SKIP]
LON_G, LAT_G = np.meshgrid(lons_s, lats_s)
ax.quiver(LON_G, LAT_G, u_s, v_s,
          transform=proj, zorder=4,
          scale=300, width=0.002, color="#08306b", alpha=0.7)

ax.scatter(PR_LON, PR_LAT, s=60, color="red", edgecolors="darkred",
           linewidths=0.8, transform=proj, zorder=5)
ax.text(PR_LON + 1.5, PR_LAT + 1.2, "CSJ", fontsize=7.5, color="darkred",
        fontweight="bold", transform=proj, zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax.set_title("ERA5 — Wind 700 hPa  (full route domain)", fontsize=10, fontweight="bold")
ax.text(0.02, 0.97, t_label, transform=ax.transAxes,
        fontsize=8, va="top", color="#333333",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

# ── Panel 3: ERA5 Specific Humidity at 850 hPa ──────────────────────────────
ax = axes[2]
basemap(ax, [rW, rE, rS, rN])

im3 = ax.pcolormesh(lon_pl, lat_pl, q850.values,
                    cmap="GnBu", vmin=0, vmax=15,
                    transform=proj, zorder=2)
plt.colorbar(im3, ax=ax, orientation="horizontal", pad=0.04,
             label="Specific humidity (g/kg)", shrink=0.85)

ax.scatter(PR_LON, PR_LAT, s=60, color="red", edgecolors="darkred",
           linewidths=0.8, transform=proj, zorder=5)
ax.text(PR_LON + 1.5, PR_LAT + 1.2, "CSJ", fontsize=7.5, color="darkred",
        fontweight="bold", transform=proj, zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax.set_title("ERA5 — Specific Humidity 850 hPa  (full route domain)", fontsize=10, fontweight="bold")
ax.text(0.02, 0.97, t_label, transform=ax.transAxes,
        fontsize=8, va="top", color="#333333",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

fig.suptitle("Sample Input Fields — 2024-12-17 18:00 UTC  (peak dust event)",
             fontsize=12, fontweight="bold", y=1.01)

plt.tight_layout()
plt.savefig("data_fields_sample.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved data_fields_sample.png")
plt.show()
