import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from pr_VIT_dustmeteo import PRAfricaDustRouteMeteoNet


# =========================================================
# 1) CONFIG LOADER
# =========================================================
def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Derived time steps
    steps_per_day        = 24 // cfg["t_step_hours"]
    cfg["t_in"]          = cfg["t_in_days"]  * steps_per_day
    cfg["t_dust"]        = cfg["t_dust_days"] * steps_per_day
    cfg["steps_per_day"] = steps_per_day

    # Coerce types
    cfg["route_patch_size"] = tuple(cfg["route_patch_size"])
    cfg["area_route"]       = list(cfg["area_route"])
    cfg["area_africa"]      = list(cfg["area_africa"])

    # Support both old single data_dir and new split era5_dir/cams_dir
    if "data_dir" in cfg and "era5_dir" not in cfg:
        cfg["era5_dir"] = cfg["data_dir"]
        cfg["cams_dir"] = cfg["data_dir"]

    return cfg


# =========================================================
# 2) FILE OPENERS
# =========================================================
def open_nc_or_zipped_nc(path, engine="netcdf4", chunks=None):
    with open(path, "rb") as f:
        sig = f.read(4)
    is_zip = sig == b"PK\x03\x04"

    if not is_zip:
        ds = xr.open_dataset(path, engine=engine, chunks=chunks)
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        return ds.sortby("time") if "time" in ds.coords else ds

    extract_dir = str(path) + "_unzipped"
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
        ds = xr.open_dataset(p, engine=engine, chunks=chunks)
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        dss.append(ds)
    ds_merged = xr.merge(dss, compat="override", join="inner")
    return ds_merged.sortby("time") if "time" in ds_merged.coords else ds_merged


# =========================================================
# 3) MULTI-FILE LOADER
# =========================================================
def load_all_files(data_dir, pattern, chunks=None):
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")
    print(f"[loader] {len(files)} file(s) for '{pattern}':")
    for f in files:
        print(f"         {f.name}")
    datasets = [open_nc_or_zipped_nc(f, chunks=chunks) for f in files]
    ds = xr.concat(datasets, dim="time").sortby("time")
    _, unique_idx = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=unique_idx)
    print(f"[loader] {len(ds.time)} steps  "
          f"({pd.Timestamp(ds.time.values[0]).date()} → {pd.Timestamp(ds.time.values[-1]).date()})")
    return ds


# =========================================================
# 4) GEOGRAPHIC HELPERS
# =========================================================
def convert_lon_if_needed(ds, west, east):
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lons = ds[lon_name].values
    if np.nanmin(lons) >= 0 and west < 0:
        west, east = west % 360, east % 360
    return west, east


def crop_area(ds, area):
    north, west, south, east = area
    lat_name = "longitude" if "longitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude"  if "latitude"  in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    west, east = convert_lon_if_needed(ds, west, east)
    ds = ds.sortby(lat_name).sortby(lon_name)
    return ds.sel({lat_name: slice(south, north), lon_name: slice(west, east)})


# =========================================================
# 5) CHANNEL ARRAY BUILDERS
# =========================================================
def build_pl_all_levels(ds_pl):
    arrays, channel_names = [], []
    for var in ["u", "v", "z", "r", "t", "q", "w", "pv"]:
        if var not in ds_pl.data_vars:
            raise ValueError(f"{var} not in ds_pl")
        for lev in ds_pl["pressure_level"].values:
            da = ds_pl[var].sel(pressure_level=lev)
            if "pressure_level" in da.coords:
                da = da.reset_coords("pressure_level", drop=True)
            ch = f"{var}_{int(lev)}"
            arrays.append(da.expand_dims(channel=[ch]))
            channel_names.append(ch)
    ds_out = xr.concat(arrays, dim="channel")
    return ds_out.transpose("time", "channel", "latitude", "longitude"), channel_names


def build_route_dataset(ds_pl, ds_sl):
    ds_pl_all, pl_channels = build_pl_all_levels(ds_pl)
    sl_vars = ["tp", "cp", "lsp", "tcwv", "blh", "msl", "v10", "u10"]
    arrays_sl = []
    for var in sl_vars:
        if var not in ds_sl.data_vars:
            raise ValueError(f"{var} not in ds_sl")
        arrays_sl.append(ds_sl[var].expand_dims(channel=[var]))
    ds_sl_out = xr.concat(arrays_sl, dim="channel").transpose("time", "channel", "latitude", "longitude")
    return xr.concat([ds_pl_all, ds_sl_out], dim="channel").transpose("time", "channel", "latitude", "longitude")


def build_dust_dataset(ds_cams):
    arr1 = ds_cams["aod550"].expand_dims(channel=["total_aod550"])
    arr2 = ds_cams["duaod550"].expand_dims(channel=["dust_aod550"])
    return xr.concat([arr1, arr2], dim="channel").transpose("time", "channel", "latitude", "longitude")


# =========================================================
# 6) AERONET TARGET
# =========================================================
def load_aeronet_targets(path, target_col="AOD_500nm-AOD", agg="median"):
    df = pd.read_csv(path, skiprows=6)
    df["datetime"] = pd.to_datetime(
        df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
        format="%d:%m:%Y %H:%M:%S"
    )
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = df[target_col].mask(df[target_col].isin([-999.0, -9999.0]))
    df = df.set_index("datetime")
    valid = df[target_col][df[target_col] > 0]
    daily = getattr(valid.resample("D"), agg)().dropna()
    print(f"[targets] {len(daily)} valid days  "
          f"({daily.index[0].date()} → {daily.index[-1].date()})")
    print(f"[targets] {agg}={daily.mean():.4f}  std={daily.std():.4f}  "
          f"baseline_MSE={daily.var():.4f}")
    return daily


# =========================================================
# 7) NORMALIZATION STATS  (compute once, cache to JSON)
# =========================================================
def compute_norm_stats(route_da, dust_da, n_samples=200, seed=42):
    """
    Estimates per-channel mean and std from n_samples random single timesteps.
    Peak RAM = one timestep at a time — never loads the full series.
    """
    rng = np.random.default_rng(seed)
    T = route_da.shape[0]
    idxs = rng.integers(0, T, size=min(n_samples, T))

    r_means, r_stds, d_means, d_stds = [], [], [], []
    for i in tqdm(idxs, desc="norm stats", unit="step"):
        xr_ = route_da.isel(time=int(i)).values.astype(np.float32)  # [C_r, H, W]
        xd_ = dust_da.isel(time=int(i)).values.astype(np.float32)   # [C_d, H, W]
        r_means.append(xr_.mean(axis=(1, 2)))
        r_stds.append(xr_.std(axis=(1, 2)))
        d_means.append(xd_.mean(axis=(1, 2)))
        d_stds.append(xd_.std(axis=(1, 2)))

    def to_t(lst):
        return torch.from_numpy(np.stack(lst).mean(axis=0).astype(np.float32)).view(1, -1, 1, 1)

    return to_t(r_means), to_t(r_stds), to_t(d_means), to_t(d_stds)


def save_norm_stats(path, route_mean, route_std, dust_mean, dust_std):
    stats = {
        "route_mean": route_mean.squeeze().tolist(),
        "route_std":  route_std.squeeze().tolist(),
        "dust_mean":  dust_mean.squeeze().tolist(),
        "dust_std":   dust_std.squeeze().tolist(),
    }
    with open(path, "w") as f:
        json.dump(stats, f)
    print(f"[norm] stats saved → {path}")


def load_norm_stats(path):
    with open(path) as f:
        stats = json.load(f)
    def to_t(key):
        return torch.tensor(stats[key], dtype=torch.float32).view(1, -1, 1, 1)
    return to_t("route_mean"), to_t("route_std"), to_t("dust_mean"), to_t("dust_std")


# =========================================================
# 8) LAZY PYTORCH DATASET
# =========================================================
class PRLazyDataset(Dataset):
    """
    If memmap files exist on scratch, uses np.memmap for O(1) index access.
    Falls back to dask-backed xarray DataArrays otherwise (slow on NFS).
    """

    def __init__(
        self,
        route_da, dust_da,
        targets, times,
        T_in, T_dust,
        route_mean=None, route_std=None,
        dust_mean=None,  dust_std=None,
        memmap_dir=None,
    ):
        self.T_in       = T_in
        self.T_dust     = T_dust
        self.route_mean = route_mean
        self.route_std  = route_std
        self.dust_mean  = dust_mean
        self.dust_std   = dust_std

        # Use memmap if available, else fall back to xarray/dask
        if memmap_dir is not None:
            route_path = Path(memmap_dir) / "route.dat"
            dust_path  = Path(memmap_dir) / "dust.dat"
            meta_path  = Path(memmap_dir) / "meta.json"
            if route_path.exists() and dust_path.exists() and meta_path.exists():
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                rs = tuple(meta["route_shape"])
                ds_ = tuple(meta["dust_shape"])
                self.route_src = np.memmap(route_path, dtype="float32", mode="r", shape=rs)
                self.dust_src  = np.memmap(dust_path,  dtype="float32", mode="r", shape=ds_)
                self.use_memmap = True
                print(f"[dataset] using memmap from {memmap_dir}")
            else:
                print(f"[dataset] memmap not found in {memmap_dir}, falling back to xarray")
                self.route_src  = route_da
                self.dust_src   = dust_da
                self.use_memmap = False
        else:
            self.route_src  = route_da
            self.dust_src   = dust_da
            self.use_memmap = False

        targets_index = pd.DatetimeIndex(targets.index).normalize()
        self.samples = []
        for i in range(len(times) - T_in):
            t_end = pd.Timestamp(times[i + T_in - 1])
            # One sample per day: window ending at 18:00 UTC
            if t_end.hour != 18:
                continue
            # Target is 7 days after the end of the route window
            t_target = t_end.normalize() + pd.Timedelta(days=7)
            if t_target in targets_index:
                self.samples.append((i, float(targets.loc[t_target])))

        print(f"[dataset] {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, y = self.samples[idx]

        if self.use_memmap:
            x_route = torch.from_numpy(
                self.route_src[i : i + self.T_in].copy()
            )
            x_dust = torch.from_numpy(
                self.dust_src[i : i + self.T_dust].copy()
            )
        else:
            x_route = torch.from_numpy(
                self.route_src.isel(time=slice(i, i + self.T_in)).values
            )
            x_dust = torch.from_numpy(
                self.dust_src.isel(time=slice(i, i + self.T_dust)).values
            )

        x_route = torch.nan_to_num(x_route, nan=0.0)
        x_dust  = torch.nan_to_num(x_dust,  nan=0.0)

        if self.route_mean is not None:
            x_route = (x_route - self.route_mean) / (self.route_std + 1e-6)
        if self.dust_mean is not None:
            x_dust = (x_dust - self.dust_mean) / (self.dust_std + 1e-6)

        return x_route, x_dust, torch.tensor(y, dtype=torch.float32)


# =========================================================
# 9) TRAINING
# =========================================================
def run_epoch(model, loader, device, optimizer=None):
    """Single train or eval pass. Pass optimizer=None for validation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n = 0.0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch_idx, (x_route, x_dust, y) in enumerate(tqdm(loader, desc="train" if is_train else "val", leave=False)):
            x_route = x_route.to(device, non_blocking=True)  # [B, T_in, C_r, H, W]
            x_dust  = x_dust.to(device, non_blocking=True)   # [B, T_dust, 2, H_d, W_d]
            y       = y.to(device, non_blocking=True)         # [B]

            y_hat = model(x_dust, x_route, return_attn=False)  # [B, 1]
            loss  = F.mse_loss(y_hat.squeeze(-1), y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                print(f"  batch {batch_idx + 1}/{len(loader)}  MSE={loss.item():.4f}")

            total_loss += loss.item() * len(y)
            n += len(y)

    return total_loss / n


# =========================================================
# 10) MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[config] loaded from {args.config}")

    # Unpack config
    ERA5_DIR        = Path(cfg["era5_dir"])
    CAMS_DIR        = Path(cfg["cams_dir"])
    AERONET_PATH    = Path(cfg["aeronet_path"])
    NORM_STATS_PATH = Path(cfg["norm_stats_path"])
    CHECKPOINT_DIR  = Path(cfg["checkpoint_dir"])
    AREA_ROUTE      = cfg["area_route"]
    AREA_AFRICA     = cfg["area_africa"]
    T_STEP_HOURS    = cfg["t_step_hours"]
    T_IN            = cfg["t_in"]
    T_DUST          = cfg["t_dust"]
    CHUNK_TIME      = cfg["chunk_time"]
    ROUTE_PATCH_SIZE = cfg["route_patch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ----------------------------------------------------------
    # Load files (dask — lazy, no data read yet)
    # ----------------------------------------------------------
    chunks = {"time": CHUNK_TIME}
    print("\n--- LOADING ERA5 PL ---")
    ds_pl = load_all_files(ERA5_DIR, cfg["pattern_era5_pl"], chunks=chunks)
    print("\n--- LOADING ERA5 SL ---")
    ds_sl = load_all_files(ERA5_DIR, cfg["pattern_era5_sl"], chunks=chunks)
    print("\n--- LOADING CAMS ---")
    ds_cams = load_all_files(CAMS_DIR, cfg["pattern_cams"], chunks=chunks)

    # ----------------------------------------------------------
    # Crop + build channel DataArrays (still lazy)
    # ----------------------------------------------------------
    ds_pl_route    = crop_area(ds_pl,   AREA_ROUTE)
    ds_sl_route    = crop_area(ds_sl,   AREA_ROUTE)
    ds_cams_africa = crop_area(ds_cams, AREA_AFRICA)

    route_da = build_route_dataset(ds_pl_route, ds_sl_route)
    dust_da  = build_dust_dataset(ds_cams_africa)

    # ----------------------------------------------------------
    # Trim spatial dims to be divisible by patch size.
    # PatchEmbed3D requires H % patch_y == 0, W % patch_x == 0.
    # ERA5 over [35N,90W,10S,30E] gives 181×481; trim to 180×480.
    # ----------------------------------------------------------
    _, py, px = ROUTE_PATCH_SIZE
    H_trim = (route_da.shape[2] // py) * py
    W_trim = (route_da.shape[3] // px) * px
    if H_trim != route_da.shape[2] or W_trim != route_da.shape[3]:
        route_da = route_da.isel(latitude=slice(0, H_trim), longitude=slice(0, W_trim))
        print(f"[spatial] route trimmed to {H_trim}×{W_trim}")

    # ----------------------------------------------------------
    # Align both to 6-hourly grid using only real timestamps
    # ----------------------------------------------------------
    all_times    = pd.DatetimeIndex(route_da.time.values)
    target_times = all_times[all_times.hour % T_STEP_HOURS == 0]
    route_da = route_da.sel(time=target_times)
    dust_da  = dust_da.sel(time=target_times)
    times    = route_da.time.values

    route_channel_names = list(route_da.coords["channel"].values)
    print(f"\nRoute: {route_da.shape}   channels: {len(route_channel_names)}")
    print(f"Dust : {dust_da.shape}")

    # ----------------------------------------------------------
    # AERONET targets
    # ----------------------------------------------------------
    targets = load_aeronet_targets(AERONET_PATH)

    # ----------------------------------------------------------
    # Norm stats — compute once, cache to JSON
    # ----------------------------------------------------------
    if NORM_STATS_PATH.exists():
        route_mean, route_std, dust_mean, dust_std = load_norm_stats(NORM_STATS_PATH)
        print(f"[norm] loaded from {NORM_STATS_PATH}")
    else:
        route_mean, route_std, dust_mean, dust_std = compute_norm_stats(
            route_da, dust_da, n_samples=cfg["norm_n_samples"]
        )
        save_norm_stats(NORM_STATS_PATH, route_mean, route_std, dust_mean, dust_std)

    # ----------------------------------------------------------
    # Dataset — time-based train / val split (no data leakage)
    # ----------------------------------------------------------
    memmap_dir = cfg.get("memmap_dir", None)
    full_dataset = PRLazyDataset(
        route_da, dust_da, targets, times, T_IN, T_DUST,
        route_mean, route_std, dust_mean, dust_std,
        memmap_dir=memmap_dir,
    )

    n_val   = max(1, int(len(full_dataset) * cfg["val_frac"]))
    n_train = len(full_dataset) - n_val
    train_ds = Subset(full_dataset, range(n_train))
    val_ds   = Subset(full_dataset, range(n_train, len(full_dataset)))
    print(f"[split] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=device.type == "cuda")

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------
    model = PRAfricaDustRouteMeteoNet(
        route_channel_names=route_channel_names,
        route_patch_size=ROUTE_PATCH_SIZE,
        embed_dim=cfg["embed_dim"],
        num_heads_space=cfg["num_heads_space"],
        num_heads_fusion=cfg["num_heads_fusion"],
        dropout=cfg["dropout"],
        out_dim=cfg["out_dim"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["num_epochs"]
    )

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    num_epochs    = cfg["num_epochs"]

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss   = run_epoch(model, val_loader,   device, optimizer=None)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{num_epochs}  "
              f"train_MSE={train_loss:.4f}  val_MSE={val_loss:.4f}  lr={lr_now:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = CHECKPOINT_DIR / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "route_channel_names": route_channel_names,
                "config": cfg,
            }, ckpt)
            print(f"           ✓ saved best checkpoint (val_MSE={val_loss:.4f})")

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.4f}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
