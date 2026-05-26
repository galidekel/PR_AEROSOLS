import matplotlib
matplotlib.use("TkAgg")  # interactive window
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

fn = "cams_eac4_total_dust_aod550_3h_202412.nc"
ds = xr.open_dataset(fn)

time_name = "valid_time"
vars2 = ["aod550", "duaod550"]   # total vs dust

# PR marker
lat_pr, lon_pr = 18.38, -65.62
if ds.longitude.max() > 180:
    lon_pr = lon_pr % 360

# fixed color scale (recommended)
vmin, vmax = 0.0, 1.5

times = pd.to_datetime(ds[time_name].values)
days = pd.to_datetime(times.date).unique()

for day in days:
    day_start = np.datetime64(day)
    day_end = day_start + np.timedelta64(1, "D")

    ds_day = ds.sel({time_name: slice(day_start, day_end)})
    t_day = pd.to_datetime(ds_day[time_name].values)
    idx = np.where(t_day.date == pd.Timestamp(day).date())[0]
    ds_day = ds_day.isel({time_name: idx})

    if ds_day.sizes[time_name] == 0:
        continue

    fig, axes = plt.subplots(8, 2, figsize=(14, 18), sharex=True, sharey=True)

    for i in range(8):
        if i >= ds_day.sizes[time_name]:
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            continue

        ts = pd.to_datetime(ds_day[time_name].values[i]).strftime("%H:%M")

        for j, v in enumerate(vars2):
            ax = axes[i, j]
            da = ds_day[v].isel({time_name: i})
            da.plot(ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax)

            ax.scatter([lon_pr], [lat_pr], c="red", s=12)
            if j == 0:
                ax.set_ylabel(ts, fontsize=10)  # hour label on left column
            ax.set_title(v if i == 0 else "")

    # one shared colorbar (based on first plotted panel)
    mappable = axes[0, 0].collections[0]
    # cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85)
    # cbar.set_label("AOD @ 550nm")

    fig.suptitle(f"Daily 3-hourly maps | {pd.Timestamp(day).strftime('%Y-%m-%d')}", fontsize=16)
    plt.tight_layout()
    plt.show(block=False)

    input("Enter for next day (Ctrl+C to stop)...")
    plt.close(fig)
