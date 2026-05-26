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
    ds_pl = ds_pl.interp(latitude=d.latitude, longitude=d.longitude)

    # crop
    d = d.sel(latitude=slice(LAT_MAX, LAT_MIN),
              longitude=slice(LON_MIN, LON_MAX))
    cams = cams.sel(latitude=slice(LAT_MAX, LAT_MIN),
                    longitude=slice(LON_MIN, LON_MAX))
    ds_pl = ds_pl.sel(latitude=slice(LAT_MAX, LAT_MIN),
                      longitude=slice(LON_MIN, LON_MAX))

    lats = d.latitude.values
    lons = d.longitude.values
    times = d.valid_time.values
    times = times[:132]

    # 12 fields
    fields = [
        lambda t: cams.duaod550.sel(valid_time=t),
        lambda t: cams.aod550.sel(valid_time=t),

        lambda t: ds_pl.u.sel(pressure_level=700).sel(valid_time=t, method="nearest"),
        lambda t: ds_pl.v.sel(pressure_level=700).sel(valid_time=t, method="nearest"),

        lambda t: ds_pl.u.sel(pressure_level=850).sel(valid_time=t, method="nearest"),
        lambda t: ds_pl.v.sel(pressure_level=850).sel(valid_time=t, method="nearest"),

        lambda t: ds_pl.z.sel(pressure_level=700).sel(valid_time=t, method="nearest"),
        lambda t: ds_pl.t.sel(pressure_level=850).sel(valid_time=t, method="nearest"),

        lambda t: ds_pl.r.sel(pressure_level=700).sel(valid_time=t, method="nearest"),

        lambda t: d.tcwv.sel(valid_time=t),
        lambda t: d.lsp_mm.sel(valid_time=t),
        lambda t: d.blh.sel(valid_time=t),
    ]
    subplot_titles = [
        "Dust AOD []", "Total AOD []",
        "U700 [m/s]", "V700 [m/s]",
        "U850 [m/s]", "V850 [m/s]",
        "Z700 [m²/s²]", "T850 [K]",
        "RH700 [%]",
        "TCWV [kg/m²]", "LSP [mm]", "BLH [m]"
    ]
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.005  # tighter vertically
    )
    # keep lat/lon aspect (no distortion)
    for i in range(1, 13):
        xref = f"x{i}" if i > 1 else "x"
        yaxis_name = f"yaxis{i}" if i > 1 else "yaxis"
        fig.update_layout({
            yaxis_name: dict(scaleanchor=xref, scaleratio=1)
        })
    for ann in fig.layout.annotations:
        ann.y = ann.y - 0.07

    # --- initial frame ---
    t0 = times[0]
    cbar_x_offsets = [0.24, 0.49, 0.74, 0.98]
    for i, field in enumerate(fields):
        r = i // 4 + 1
        c = i % 4 + 1

        z = field(t0).values

        # --- LSP handling ---
        if i == 10:  # LSP index
            z = np.clip(z, 0, 3)
            zmin, zmax = 0, 3  # 🔥 FIX SCALE (CRITICAL)
            cmap = "Viridis"  # 🔥 better for precipitation
        elif i==0:
            z = np.clip(z, 0, 0.1)
            zmin, zmax = 0, 0.1  # 🔥 FIX SCALE (CRITICAL)
            cmap = "Viridis"  # 🔥 better for precipitation
        else:
            zmin, zmax = None, None
            cmap = "Viridis"

        axis_id = i + 1
        xaxis_name = f"xaxis{axis_id}" if axis_id > 1 else "xaxis"
        yaxis_name = f"yaxis{axis_id}" if axis_id > 1 else "yaxis"

        x_domain = fig.layout[xaxis_name].domain
        y_domain = fig.layout[yaxis_name].domain

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=lons,
                y=lats,
                colorscale=cmap,
                zmin=zmin,  # 🔥 applied only for LSP
                zmax=zmax,
                showscale=True,
                colorbar=dict(
                    x=x_domain[1] + 0.005 * (1 - x_domain[1]),
                    y=(y_domain[0] + y_domain[1]) / 2,
                    len=(y_domain[1] - y_domain[0]) * 0.8,
                    thickness=6,
                    outlinewidth=0
                )
            ),
            row=r, col=c
        )
    fig.update_layout(
        margin=dict(l=20, r=250, t=40, b=20)
    )
    for i in range(1, 13):
        xaxis_name = f"xaxis{i}" if i > 1 else "xaxis"
        yaxis_name = f"yaxis{i}" if i > 1 else "yaxis"

        fig.update_layout({
            xaxis_name: dict(
                range=[LON_MIN, LON_MAX],
                constrain="domain",
                showgrid=False
            ),
            yaxis_name: dict(
                range=[LAT_MAX, LAT_MIN],  # note inverted for lat
                constrain="domain",
                showgrid=False
            )
        })
    fig.update_xaxes(title_text="Longitude [°]", row=1, col=1)
    fig.update_yaxes(title_text="Latitude [°]", row=1, col=1)
    # --- frames ---
    frames = []
    for t in times:

        frame_data = []
        for field in fields:
            z = field(t).values
            if i == 10:
                z = np.clip(z, 0, 3)
            if i == 0:
                z = np.clip(z, 0, 0.1)
            frame_data.append(
                go.Heatmap(
                    z=z
                    # x=lons,
                    # y=lats,
                    # colorscale="Viridis",
                    # showscale=False
                )
            )

        frames.append(go.Frame(data=frame_data, name=str(t)))

    fig.frames = frames
    add_coastlines(fig, lons, lats, rows=3, cols=4)
    # Puerto Rico location
    pr_lon, pr_lat = -66.5, 18.2
    pr_lon, pr_lat = pr_lon + 2, pr_lat +0.7


    # Arrow start (Atlantic)
    start_lon, start_lat = -55, 22

    for i in range(1, 13):
        xref = f"x{i}" if i > 1 else "x"
        yref = f"y{i}" if i > 1 else "y"

        fig.add_annotation(
            x=pr_lon,
            y=pr_lat,
            ax=start_lon,
            ay=start_lat,
            xref=xref,
            yref=yref,
            axref=xref,
            ayref=yref,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="white",
            opacity=0.9
        )
    # slider
    fig.update_layout(
        sliders=[{
            "steps": [
                {"method": "animate",
                 "args": [[str(t)], {"mode": "immediate"}],
                 "label": str(t)[:16]}
                for t in times
            ]
        }]
    )

    fig.write_html(out_html)
    print("saved:", out_html)
# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    ds = open_era5_zip(ZIP_PATH)
    make_dashboard(ds)