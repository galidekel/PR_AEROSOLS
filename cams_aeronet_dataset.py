import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os, zipfile

# =========================================================
# CONFIG
# =========================================================
AREA_ROUTE  = [35, -90, -10, 30]   # [N, W, S, E]  — full transport corridor
AREA_AFRICA = [35, -17, -10, 30]   # [N, W, S, E]  — Africa dust source only

# Temporal: subsample to 6-hourly (4 steps/day)
STEPS_PER_DAY  = 4
T_IN_DAYS      = 7               # ERA5 route window: 7 days → 28 steps
T_IN           = T_IN_DAYS * STEPS_PER_DAY   # 28 steps
T_DUST_DAYS    = 1               # CAMS dust source: 1 day at t → 4 steps
T_DUST         = T_DUST_DAYS * STEPS_PER_DAY

# Spatial: coarsen ERA5 0.25° → 0.5°, then trim to patch-divisible size
# patch_size=(1,20,20) → H and W must be divisible by 20
# At 0.5°: domain gives 91 lat × 241 lon → trim to 80 × 240
# (lat [35N → 4.5S], lon [-90W → 29.5E])
COARSEN_FACTOR = 2      # 0.25° × 2 = 0.5°
H_TARGET       = 80     # lat points after trim
W_TARGET       = 240    # lon points after trim

TARGET = "AE_440_870"   # Ångström Exponent 440–870 nm at Cape San Juan, t+7
def open_nc_or_zipped_nc(path, engine="netcdf4"):
    """
    Opens either:
      - a normal NetCDF file, or
      - a ZIP archive (even if it ends with .nc) that contains one or more .nc files.
    If ZIP contains multiple .nc files, they are merged on common coords (time/lat/lon).
    """
    # Detect ZIP by signature
    with open(path, "rb") as f:
        sig = f.read(4)
    is_zip = sig == b"PK\x03\x04"

    if not is_zip:
        ds = xr.open_dataset(path, engine=engine)
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        return ds.sortby("time") if "time" in ds.coords else ds

    # ZIP case: extract and open all inner .nc
    extract_dir = path + "_unzipped"
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)
        nc_paths = [
            os.path.join(extract_dir, name)
            for name in zf.namelist()
            if name.endswith(".nc")
        ]

    dss = []
    for p in nc_paths:
        ds = xr.open_dataset(p, engine=engine)
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        dss.append(ds)

    ds_merged = xr.merge(dss, compat="override", join="inner")
    return ds_merged.sortby("time") if "time" in ds_merged.coords else ds_merged

# -----------------------------
# 1) Load gridded data
# -----------------------------
def crop_area(ds, area):
    """
    area = [N, W, S, E]
    """

    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"

    N, W, S, E = area

    lat = ds[lat_name]

    # ERA5 lat is usually descending
    if lat[0] > lat[-1]:
        ds = ds.sel({lat_name: slice(N, S)})
    else:
        ds = ds.sel({lat_name: slice(S, N)})

    ds = ds.sel({lon_name: slice(W, E)})

    return ds
def build_route_dataset(ds_pl, ds_sl):
    import xarray as xr
    lat_name = "latitude" if "latitude" in ds_pl.dims else "lat"
    lon_name = "longitude" if "longitude" in ds_pl.dims else "lon"

    lat = ds_pl[lat_name].values
    lon = ds_pl[lon_name].values

    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])

    print("Resolution of pl vars:")
    print("dlat =", dlat)
    print("dlon =", dlon)
    lat_name = "latitude" if "latitude" in ds_sl.dims else "lat"
    lon_name = "longitude" if "longitude" in ds_sl.dims else "lon"

    lat = ds_sl[lat_name].values
    lon = ds_sl[lon_name].values

    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])

    print("Resolution of Sl vars:")
    print("dlat =", dlat)
    print("dlon =", dlon)


    da_list = []
    ch_names = []

    # ---- detect dims
    level_dim = "level" if "level" in ds_pl.dims else "pressure_level"

    lat_name = "latitude" if "latitude" in ds_pl.dims else "lat"
    lon_name = "longitude" if "longitude" in ds_pl.dims else "lon"

    levels = ds_pl[level_dim].values

    # -----------------------------------------------------
    # PL → (var × level) → channels
    # -----------------------------------------------------
    level_dim = "level" if "level" in ds_pl.dims else "pressure_level"
    levels = ds_pl[level_dim].values

    for v in ds_pl.data_vars:
        for lev in levels:
            da = ds_pl[v].sel({level_dim: lev}).drop_vars(level_dim)
            da_list.append(da)
            ch_names.append(f"{v}_{int(lev)}")

    # -----------------------------------------------------
    # SL → channels
    # -----------------------------------------------------
    for v in ds_sl.data_vars:
        da_list.append(ds_sl[v])
        ch_names.append(v)

    # -----------------------------------------------------
    # stack into channel dim
    # -----------------------------------------------------
    ds_out = xr.concat(da_list, dim="channel")
    ds_out = ds_out.assign_coords(channel=ch_names)

    ds_out = ds_out.transpose("time", "channel", lat_name, lon_name)

    return ds_out, ch_names
def build_dust_dataset(ds_cams):
    import xarray as xr
    lat_name = "latitude" if "latitude" in ds_cams.dims else "lat"
    lon_name = "longitude" if "longitude" in ds_cams.dims else "lon"

    lat = ds_cams[lat_name].values
    lon = ds_cams[lon_name].values

    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])

    print("Resolution:")
    print("dlat =", dlat)
    print("dlon =", dlon)
    da_list = []
    ch_names = []

    # ---- detect lat/lon names
    lat_name = "latitude" if "latitude" in ds_cams.dims else "lat"
    lon_name = "longitude" if "longitude" in ds_cams.dims else "lon"

    # -----------------------------------------------------
    # Build channels (each variable = one channel)
    # -----------------------------------------------------
    for v in ds_cams.data_vars:
        da = ds_cams[v]

        # ---- drop any extra coordinates (safety)
        for coord in list(da.coords):
            if coord not in ["time", lat_name, lon_name]:
                da = da.drop_vars(coord, errors="ignore")

        da_list.append(da)
        ch_names.append(v)

    # -----------------------------------------------------
    # Stack into channel dimension
    # -----------------------------------------------------
    ds_out = xr.concat(da_list, dim="channel")
    ds_out = ds_out.assign_coords(channel=ch_names)

    # -----------------------------------------------------
    # Enforce dimension order
    # -----------------------------------------------------
    ds_out = ds_out.transpose("time", "channel", lat_name, lon_name)

    return ds_out, ch_names
def coarsen_spatial(ds, factor=COARSEN_FACTOR):
    """Coarsen ERA5 0.25° → 0.5° by averaging over (factor × factor) blocks."""
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    return ds.coarsen({lat_name: factor, lon_name: factor}, boundary="trim").mean()


def trim_to_target(arr, h=H_TARGET, w=W_TARGET):
    """Trim spatial dims of numpy array [..., H, W] to [h, w] from top-left."""
    return arr[..., :h, :w]


def load_grids(fp_pl, fp_sl, fp_cams):
    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------
    ds_pl = open_nc_or_zipped_nc(fp_pl)
    ds_sl = open_nc_or_zipped_nc(fp_sl)
    ds_c  = open_nc_or_zipped_nc(fp_cams)

    print("\n--- RAW TIME ---")
    print("PL:", len(ds_pl.time))
    print("SL:", len(ds_sl.time))
    print("CAMS:", len(ds_c.time))

    # -----------------------------------------------------
    # Crop to domain
    # -----------------------------------------------------
    ds_pl_route = crop_area(ds_pl, AREA_ROUTE)
    ds_sl_route = crop_area(ds_sl, AREA_ROUTE)
    ds_c_africa = crop_area(ds_c, AREA_AFRICA)

    # -----------------------------------------------------
    # Build channel datasets
    # -----------------------------------------------------
    ds_route, route_channels = build_route_dataset(ds_pl_route, ds_sl_route)
    ds_dust,  dust_channels  = build_dust_dataset(ds_c_africa)

    print("\n--- AFTER BUILD ---")
    print("Route:", ds_route.shape)
    print("Dust :", ds_dust.shape)

    # -----------------------------------------------------
    # Align time (intersection)
    # -----------------------------------------------------
    common_time = np.intersect1d(ds_route.time.values, ds_dust.time.values)
    ds_route = ds_route.sel(time=common_time)
    ds_dust  = ds_dust.sel(time=common_time)

    # -----------------------------------------------------
    # Subsample to 6-hourly (every 2nd step if data is 3h)
    # -----------------------------------------------------
    dt_ns = int(ds_route.time.values[1] - ds_route.time.values[0])
    dt_h  = dt_ns / 3_600_000_000_000   # nanoseconds → hours
    if dt_h < (24 / STEPS_PER_DAY):
        step = int((24 / STEPS_PER_DAY) / dt_h)
        ds_route = ds_route.isel(time=slice(None, None, step))
        ds_dust  = ds_dust.isel(time=slice(None, None, step))
        print(f"\n--- SUBSAMPLED TO 6h (every {step} steps) ---")
        print("Route time:", len(ds_route.time))

    # -----------------------------------------------------
    # Coarsen ERA5 route to 0.5° (CAMS stays at native 0.75° over Africa)
    # -----------------------------------------------------
    ds_route = coarsen_spatial(ds_route)

    print("\n--- AFTER COARSEN (0.5°) ---")
    print("Route:", ds_route.shape)
    print("Dust :", ds_dust.shape)

    # -----------------------------------------------------
    # Convert to numpy:
    #   route_all → trim to patch-divisible 80×240
    #   dust_all  → keep native CAMS dims over Africa (~61 lat × 67 lon at 0.75°)
    # -----------------------------------------------------
    route_all = trim_to_target(ds_route.values.astype(np.float32))   # [T, C_r, 80, 240]
    dust_all  = ds_dust.values.astype(np.float32)                     # [T,  2,  H_d, W_d]
    times     = ds_route.time.values

    print("\n--- FINAL ARRAYS ---")
    print("route_all:", route_all.shape)   # [T, 48, 80, 240]
    print("dust_all :", dust_all.shape)    # [T,  2, ~61, ~67]

    return route_all, dust_all, times, route_channels, dust_channels
# -----------------------------
# 2) Load AERONET and make a 3-hourly time series
# -----------------------------

def load_aeronet_lev20(fp_aeronet, targets=TARGETS, tz="UTC", time_index=None):
    import pandas as pd

    # -----------------------------------------------------
    # Find header
    # -----------------------------------------------------
    with open(fp_aeronet, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines[:200]):
        l = line.lower()
        if ("date" in l) and ("time" in l) and ("," in line):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find AERONET header line.")

    # -----------------------------------------------------
    # Load table
    # -----------------------------------------------------
    df = pd.read_csv(fp_aeronet, skiprows=header_idx, low_memory=False)

    # -----------------------------------------------------
    # Identify date/time columns
    # -----------------------------------------------------
    date_col = next((c for c in df.columns if "date(" in c.lower()), None)
    time_col = next((c for c in df.columns if "time(" in c.lower()), None)

    if date_col is None or time_col is None:
        raise ValueError(f"Could not find Date/Time columns.")

    # -----------------------------------------------------
    # Parse datetime
    # -----------------------------------------------------
    dt = pd.to_datetime(
        df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
        format="%d:%m:%Y %H:%M:%S",
        errors="raise",
    )

    df = (
        df.assign(datetime=dt)
          .dropna(subset=["datetime"])
          .set_index("datetime")
          .sort_index()
    )

    # -----------------------------------------------------
    # Convert + select targets
    # -----------------------------------------------------
    for c in targets:
        if c not in df.columns:
            raise ValueError(f"Missing target column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[targets]

    # -----------------------------------------------------
    # Localize timezone
    # -----------------------------------------------------
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)

    # -----------------------------------------------------
    # Optional alignment (NO averaging)
    # -----------------------------------------------------
    if time_index is not None:
        df = df.reindex(time_index, method="nearest", tolerance="1H")

    return df

# -----------------------------
# 3) Build sample index pairs: t -> t+7d (only where y exists)
# -----------------------------
def build_pairs(ds_grid, df_aero, lead_days=0, T_in_days=T_IN_DAYS):
    """
    Sliding-window sample builder.

    Returns
    -------
    X_route : [N, T_in, C_r, H, W]
    X_dust  : [N, T_in, 2,   H, W]
    Y       : [N, n_targets]
    """
    route_all, dust_all = ds_grid[0], ds_grid[1]

    T_total = route_all.shape[0]
    T_in    = T_in_days * STEPS_PER_DAY          # 28
    T_lead  = lead_days * STEPS_PER_DAY
    T_tgt   = STEPS_PER_DAY                      # average over 1 target day

    X_route_list, X_dust_list, Y_list = [], [], []

    for t0 in range(T_total - T_in - T_lead - T_tgt):
        t_start = t0
        t_end   = t0 + T_in

        x_r = route_all[t_start:t_end]           # [T_in, C_r, H, W]
        x_d = dust_all[t_start:t_end]            # [T_in, 2,   H, W]

        # target: mean over the T_tgt steps immediately after the lead
        t_tgt_start = t_end + T_lead
        t_tgt_end   = t_tgt_start + T_tgt

        y_window = df_aero.iloc[t_tgt_start:t_tgt_end]

        if y_window.isna().all().all():
            continue

        y = y_window.mean(axis=0).values
        if np.isnan(y).any():
            continue

        X_route_list.append(x_r)
        X_dust_list.append(x_d)
        Y_list.append(y)

    return np.array(X_route_list), np.array(X_dust_list), np.array(Y_list)
# =========================================================
# 4) Dataset — compatible with PRAfricaDustRouteMeteoNet
# =========================================================
class PRNetDataset(Dataset):
    """
    PyTorch Dataset for PRAfricaDustRouteMeteoNet.

    Each sample:
      x_dust  : [T_in, 2,   H, W]  float32
      x_route : [T_in, C_r, H, W]  float32
      y       : [n_targets]         float32

    Normalization uses per-channel statistics computed from the
    training split (pass mean/std arrays; None = no normalization).
    """
    def __init__(
        self,
        X_route,          # [N, T, C_r, H, W]  numpy
        X_dust,           # [N, T, 2,   H, W]  numpy
        Y,                # [N, L]              numpy
        route_mean=None,  # [1, 1, C_r, 1, 1]
        route_std=None,
        dust_mean=None,   # [1, 1, 2,   1, 1]
        dust_std=None,
        y_mean=None,      # [1, L]
        y_std=None,
        normalize_y=False,
    ):
        assert len(X_route) == len(X_dust) == len(Y)
        self.X_route = X_route
        self.X_dust  = X_dust
        self.Y       = Y

        self.route_mean = route_mean
        self.route_std  = route_std
        self.dust_mean  = dust_mean
        self.dust_std   = dust_std
        self.y_mean     = y_mean
        self.y_std      = y_std
        self.normalize_y = normalize_y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x_r = self.X_route[idx].astype(np.float32)   # [T, C_r, H, W]
        x_d = self.X_dust[idx].astype(np.float32)    # [T, 2,   H, W]
        y   = self.Y[idx].astype(np.float32)         # [L]

        if self.route_mean is not None and self.route_std is not None:
            x_r = (x_r - self.route_mean) / (self.route_std + 1e-8)
        if self.dust_mean is not None and self.dust_std is not None:
            x_d = (x_d - self.dust_mean) / (self.dust_std + 1e-8)
        if self.normalize_y and self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / (self.y_std + 1e-8)

        return {
            "x_route": torch.from_numpy(x_r),   # [T, C_r, H, W]
            "x_dust":  torch.from_numpy(x_d),   # [T, 2,   H, W]
            "y":       torch.from_numpy(y),      # [L]
        }


def compute_norm_stats(X_route, X_dust):
    """
    Compute per-channel mean/std from training arrays.
    Returns 4D arrays [1, C, 1, 1] — ready to broadcast against a
    single sample [T, C, H, W] inside __getitem__.
    """
    # collapse N, T, H, W → keep only C axis
    route_mean = X_route.mean(axis=(0, 1, 3, 4))[:, None, None].astype(np.float32)  # [C_r, 1, 1]
    route_std  = X_route.std(axis=(0, 1, 3, 4))[:, None, None].astype(np.float32)
    dust_mean  = X_dust.mean(axis=(0, 1, 3, 4))[:, None, None].astype(np.float32)   # [2, 1, 1]
    dust_std   = X_dust.std(axis=(0, 1, 3, 4))[:, None, None].astype(np.float32)
    return route_mean, route_std, dust_mean, dust_std


# =========================================================
# 5) Example wiring
# =========================================================
if __name__ == "__main__":
    fp_pl   = "era5_pl_250_500_700_850_900_u_v_z_rh_t_q_w_pv_3h_202412.nc"
    fp_sl   = "era5_sl_mslp_tp_cp_lsp_blh_u10_v10_tcwv_3h_202412.nc"
    fp_cams = "cams_eac4_total_dust_aod550_3h_202412.nc"
    fp_aero = "PR AEROSOLS DATA/OneDrive_2_02-12-2025/AERONET_20040101_20251231_Cape_San_Juan/20040101_20251231_Cape_San_Juan.tot_lev20"

    # --- Load grids (6h subsampled, 0.5°, 80×240) ---
    route_all, dust_all, times, route_channels, dust_channels = load_grids(fp_pl, fp_sl, fp_cams)

    # --- Load AERONET targets ---
    df_aero = load_aeronet_lev20(fp_aero, tz="UTC")

    # --- Build sliding-window samples ---
    X_route, X_dust, Y = build_pairs(
        (route_all, dust_all), df_aero, lead_days=0
    )

    print("\n--- SAMPLE SHAPES ---")
    print(f"X_route : {X_route.shape}")   # [N, 28, 48, 80, 240]
    print(f"X_dust  : {X_dust.shape}")    # [N, 28,  2, ~61, ~67]  (Africa, native 0.75°)
    print(f"Y       : {Y.shape}")         # [N, 3]
    mb = (X_route.nbytes + X_dust.nbytes) / 1e6 / len(X_route)
    print(f"Memory per sample: {mb:.1f} MB")

    # --- Train/val split (80/20 chronological) ---
    n = len(Y)
    n_train = int(0.8 * n)
    X_route_tr, X_dust_tr, Y_tr = X_route[:n_train], X_dust[:n_train], Y[:n_train]
    X_route_vl, X_dust_vl, Y_vl = X_route[n_train:], X_dust[n_train:], Y[n_train:]

    # --- Normalization stats from train split only ---
    route_mean, route_std, dust_mean, dust_std = compute_norm_stats(X_route_tr, X_dust_tr)

    train_ds = PRNetDataset(
        X_route_tr, X_dust_tr, Y_tr,
        route_mean=route_mean, route_std=route_std,
        dust_mean=dust_mean,   dust_std=dust_std,
    )
    val_ds = PRNetDataset(
        X_route_vl, X_dust_vl, Y_vl,
        route_mean=route_mean, route_std=route_std,
        dust_mean=dust_mean,   dust_std=dust_std,
    )

    sample = train_ds[0]
    print(f"\n--- DATASET CHECK ---")
    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"x_route shape : {sample['x_route'].shape}")
    print(f"x_dust  shape : {sample['x_dust'].shape}")
    print(f"y       shape : {sample['y'].shape}")
    print(f"Route channels: {route_channels}")
