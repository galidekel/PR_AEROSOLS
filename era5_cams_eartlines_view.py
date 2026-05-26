import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd


ZIP_PATH  = Path("era5_sl_mslp_tp_cp_lsp_blh_u10_v10_tcwv_3h_202412.nc")
CAMS_PATH = Path("cams_eac4_total_dust_aod550_3h_202412.nc")
PL_PATH = Path("era5_pl_250_500_700_850_900_u_v_z_rh_t_q_w_pv_3h_202412.nc")


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
# Downsample
# -------------------------------
def downsample(ds, step_lat=2, step_lon=2):
    return ds.isel(
        latitude=slice(None, None, step_lat),
        longitude=slice(None, None, step_lon),
    )


# -------------------------------
# Coastlines overlay
# -------------------------------
import requests
import os
import json

def load_coastline_geojson():
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    fname = "coastlines.geojson"

    if not os.path.exists(fname):
        print("Downloading coastlines...")
        r = requests.get(url)
        with open(fname, "w") as f:
            f.write(r.text)

    with open(fname) as f:
        return json.load(f)


def add_coastlines(fig, lons, lats, rows=4, cols=3):

    import requests, json, os
    import plotly.graph_objects as go

    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    fname = "coastlines.geojson"

    if not os.path.exists(fname):
        print("Downloading coastlines...")
        r = requests.get(url)
        with open(fname, "w") as f:
            f.write(r.text)

    with open(fname) as f:
        geo = json.load(f)

    lon_min, lon_max = -90, 30
    lat_min, lat_max = -10, 35

    for i in range(1, rows * cols + 1):

        xref = f"x{i}" if i > 1 else "x"
        yref = f"y{i}" if i > 1 else "y"

        xaxis_key = f"xaxis{i}" if i > 1 else "xaxis"
        yaxis_key = f"yaxis{i}" if i > 1 else "yaxis"

        # set axis limits correctly
        fig.update_layout({
            xaxis_key: dict(range=[lon_min, lon_max]),
            yaxis_key: dict(range=[lat_min, lat_max])
        })

        for feature in geo["features"]:
            geom = feature["geometry"]

            def draw(coords):
                xs = []
                ys = []

                for lon, lat in coords:
                    if (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max):
                        xs.append(lon)
                        ys.append(lat)
                    else:
                        xs.append(None)  # breaks line
                        ys.append(None)

                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="black", width=0.5),
                    showlegend=False,
                    xaxis=xref,
                    yaxis=yref
                ))
            if geom["type"] == "Polygon":
                for coords in geom["coordinates"]:
                    draw(coords)

            elif geom["type"] == "MultiPolygon":
                for poly in geom["coordinates"]:
                    for coords in poly:
                        draw(coords)
# Dashboard
# -------------------------------
def make_dashboard(ds_era5, out_html="dashboard.html"):

    # -------------------------------
    # REGION (NEW)
    # -------------------------------
    LAT_MIN, LAT_MAX = -10, 35
    LON_MIN, LON_MAX = -90, 30

    era5_vars = ["msl_hpa", "u10", "v10", "blh",
                 "tcwv", "tp_mm", "cp_mm", "lsp_mm"]

    d = downsample(ds_era5[era5_vars])

    # CAMS
    ds_cams = xr.open_dataset(CAMS_PATH)
    cams = ds_cams[["aod550", "duaod550"]]
    cams = cams.interp(valid_time=d.valid_time,
                       latitude=d.latitude,
                       longitude=d.longitude)

    ds_pl = xr.open_dataset(PL_PATH)

    ds_pl = ds_pl.interp(
        latitude=d.latitude,
        longitude=d.longitude
    )

    ds_pl = ds_pl.sel(valid_time=d.valid_time, method="nearest")

    # -------------------------------
    # 🔥 CROP DATA (NEW)
    # -------------------------------
    d = d.sel(latitude=slice(LAT_MAX, LAT_MIN),
              longitude=slice(LON_MIN, LON_MAX))

    cams = cams.sel(latitude=slice(LAT_MAX, LAT_MIN),
                    longitude=slice(LON_MIN, LON_MAX))

    ds_pl = ds_pl.sel(latitude=slice(LAT_MAX, LAT_MIN),
                      longitude=slice(LON_MIN, LON_MAX))

    lats = d.latitude.values
    lons = d.longitude.values
    times = d.valid_time.values

    # -------------------------------
    # Subplots
    # -------------------------------
    subplot_titles = [
        "Dust AOD", "Total AOD",
        "U700", "V700",
        "U850", "V850",
        "Z700", "T850",
        "RH700",
        "TCWV", "LSP", "BLH"
    ]

    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.02
    )
    for ann in fig.layout.annotations:
        ann.y = ann.y - 0.05  # 👈 move titles DOWN (closer to panels)
    # -------------------------------
    # Fix aspect ratio
    # -------------------------------
    for i in range(1, 13):
        x_ref = f"x{i}" if i > 1 else "x"
        yaxis_name = f"yaxis{i}" if i > 1 else "yaxis"

        fig.update_layout({
            yaxis_name: dict(scaleanchor=x_ref, scaleratio=1)
        })

    # -------------------------------
    # Fields
    # -------------------------------
    fields = [
        cams.duaod550,
        cams.aod550,

        ds_pl.u.sel(pressure_level=700, method="nearest"),
        ds_pl.v.sel(pressure_level=700, method="nearest"),

        ds_pl.u.sel(pressure_level=850, method="nearest"),
        ds_pl.v.sel(pressure_level=850, method="nearest"),

        ds_pl.z.sel(pressure_level=700, method="nearest"),
        ds_pl.t.sel(pressure_level=850, method="nearest"),

        ds_pl.r.sel(pressure_level=700, method="nearest"),

        d.tcwv,
        d.lsp_mm,
        d.blh
    ]

    # -------------------------------
    # Scales
    # -------------------------------
    scales = {
        0: (0, 0.1),
        1: (0, 1.5),
        2: (-20, 20),
        3: (-20, 20),
        4: (-20, 20),
        5: (-20, 20),
        6: (29000, 32000),
        7: (260, 310),
        8: (0, 100),
        9: (0, 70),
        10: (0, 10),
        11: (0, 2000)
    }

    t0 = 0

    # -------------------------------
    # Plot
    # -------------------------------
    for i, field in enumerate(fields):
        r = i // 4 + 1
        c = i % 4 + 1

        z = field.isel(valid_time=t0).values
        zmin, zmax = scales[i]

        axis_id = i + 1
        xaxis_name = f"xaxis{axis_id}" if axis_id > 1 else "xaxis"
        yaxis_name = f"yaxis{axis_id}" if axis_id > 1 else "yaxis"

        x_domain = fig.layout[xaxis_name].domain
        y_domain = fig.layout[yaxis_name].domain

        x_pos = x_domain[1] + 0.01
        y_pos = (y_domain[0] + y_domain[1]) / 2

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=lons,
                y=lats,
                colorscale="Viridis",
                zmin=zmin,
                zmax=zmax,
                showscale=True,
                colorbar=dict(
                    len=(y_domain[1] - y_domain[0]) * 0.6,
                    thickness=8,
                    x=x_pos,
                    y=y_pos
                )
            ),
            row=r, col=c
        )

    # -------------------------------
    # Coastlines (UPDATED)
    # -------------------------------
    add_coastlines(fig, lons, lats, rows=3, cols=4)

    fig.update_layout(
        height=700,  # 👈 smaller height = tighter rows
        width=1700,
    )
    for i in range(1, 13):
        xaxis_name = f"xaxis{i}" if i > 1 else "xaxis"
        yaxis_name = f"yaxis{i}" if i > 1 else "yaxis"

        fig.update_layout({
            xaxis_name: dict(
                range=[-90, 30],
                constrain="domain",  # 🔥 KEY
                showgrid=False  # remove grid outside
            ),
            yaxis_name: dict(
                range=[35, -10],
                constrain="domain",  # 🔥 KEY
                showgrid=False
            )
        })
    fig.write_html(out_html)
    print(f"Saved to {out_html}")

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    ds = open_era5_zip(ZIP_PATH)
    make_dashboard(ds)