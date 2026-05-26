# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep-learning research project predicting aerosol optical depth (AOD) at Puerto Rico's Cape San Juan AERONET station from African dust transport. The model is trained on two gridded reanalysis sources:
- **ERA5** (ECMWF): pressure-level (PL) and single-level (SL) meteorological fields at 3-hourly resolution
- **CAMS EAC4** (Copernicus): total and dust AOD at 550 nm, 3-hourly

The study domain spans Africa → Atlantic → Caribbean: `[35°N, 90°W, 10°S, 30°E]` (N, W, S, E). The Africa dust source sub-region is `[35°N, 20°W, 10°S, 30°E]`.

## Environment

The project uses a local venv at `.venv/` (Python 3.12). Activate it before running anything:

```bash
source .venv/bin/activate
```

Key packages: `torch`, `xarray`, `numpy`, `pandas`, `netCDF4`, `cartopy`, `matplotlib`, `cdsapi`, `scipy`, `geopandas`.

## Data Download

Scripts are designed to run on a remote HPC server (output path: `/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS/`). Locally the `.nc` files live in the project root.

```bash
# ERA5 (CDS API required, ~/.cdsapirc must be configured)
python download_era5_data.py                  # all years 2004–2024
python download_era5_data.py --year 2024      # single year
python download_era5_data.py --year 2024 --month 12  # single month

# CAMS EAC4 (ADS API; set CDSAPI_URL + CDSAPI_KEY env vars)
python download_cams_data.py

# Quick API connectivity tests
python test_era5.py
python test_cams.py
```

NetCDF files may arrive as ZIP archives with a `.nc` extension. The helper `open_nc_or_zipped_nc()` (defined in both `pr_dataset.py` and `cams_aeronet_dataset.py`) detects the ZIP magic bytes and extracts automatically.

## Running the Pipeline

```bash
# Inspect raw AERONET data and plot AOD time series
python main.py

# Build grid arrays from local sample .nc files (prints shapes, no training)
python pr_dataset.py         # uses 200409 sample files
python cams_aeronet_dataset.py  # uses 202412 sample files + AERONET

# Visualization: 3-hourly CAMS maps (interactive, one day at a time)
python visualize.py

# Cartopy dust-transport schematic
python cart.py

# CSJ in-situ aerosol optical properties summary
python cjs_data.py
```

## Architecture

### Models

**`PR_VIT.py` — `PRViT`** (single-timestamp input)
- Input: `[B, C, H, W]` — all channels stacked for one time step
- One `PatchEmbed` (Conv2d) per variable group → spatial patch tokens
- Gate 1 (`CrossAttentionGate`): each learnable variable token attends to its own spatial patches → variable summary `[B, Nv, D]`
- Gate 2: a learnable PR token attends to all variable summaries → scalar embedding `[B, D]`
- MLP regression head → `[B, out_dim]` AOD vector

**`pr_VIT_dustmeteo.py` — `PRAfricaDustRouteMeteoNet`** (sequence input, main model)
- Dual-branch architecture for temporal sequences `(T=48 steps, 3h resolution ≈ 6-day history)`:
  - `DustSourceEncoder`: 3D CNN (`Conv3d`) on Africa CAMS dust patches `[B, T, 2, H_d, W_d]` → `[B, D]`
  - `RouteMeteoEncoder`: one `PatchEmbed3D` per meteorological channel, each channel token attends to its spatiotemporal patches `[B, T, C_r, H_r, W_r]` → `[B, C_r, D]`
- Fusion: a PR token attends (cross-attention) to the concatenated dust + route summaries → `[B, D]`
- MLP head → `[B, out_dim]`
- All attention gates include pre-norm + residual + MLP (post-norm)

### Data Pipeline (`cams_aeronet_dataset.py`, `pr_dataset.py`)

1. **Load**: `open_nc_or_zipped_nc()` opens NetCDF or ZIP-wrapped NetCDF; normalises the time coordinate (`valid_time` → `time`).
2. **Crop**: `crop_area(ds, [N, W, S, E])` handles descending ERA5 latitudes and 0–360 vs −180–180 longitude conventions.
3. **Build channel arrays**:
   - `build_route_dataset(ds_pl, ds_sl)`: flattens `(variable × pressure_level)` into a channel dim + SL variables → `[T, C_r, H, W]`
   - `build_dust_dataset(ds_cams)`: `aod550` + `duaod550` → `[T, 2, H, W]`
4. **Time alignment**: CAMS is interpolated onto ERA5 timestamps (or strict intersection for training).
5. **AERONET targets** (`load_aeronet_lev20()`): parses the `.tot_lev20` ASCII format (6-row header), converts to UTC DatetimeIndex, replaces −999/−9999 flags with NaN.
6. **Sample pairs** (`build_pairs()`): sliding window — 7-day input → target AOD window at a configurable lead.
7. **Dataset**: `PRAfricaDustRouteDataset` (PyTorch `Dataset`) applies per-channel z-score normalisation at item retrieval.

### Target variable

AERONET level-2.0 AOD at wavelengths `440 nm`, `500 nm`, `870 nm` (Cape San Juan station, `18.38°N, 65.62°W`).

### ERA5 channels (pressure-level variables)

`u`, `v`, `z`, `r`, `t`, `q`, `w`, `pv` at levels 250, 500, 700, 850, 900 hPa plus SL: `tp`, `cp`, `lsp`, `tcwv`, `blh`, `msl`, `u10`, `v10`.

## Key File Map

| File | Role |
|------|------|
| `pr_VIT_dustmeteo.py` | Main dual-branch model (`PRAfricaDustRouteMeteoNet`) |
| `PR_VIT.py` | Single-frame variant (`PRViT`) |
| `cams_aeronet_dataset.py` | Full data pipeline + `GridsToAODLeadDataset` |
| `pr_dataset.py` | Alternative dataset class (`PRAfricaDustRouteDataset`) |
| `train_main.py` | Training entry point (WIP) |
| `download_era5_data.py` | ERA5 bulk download via CDS API |
| `download_cams_data.py` | CAMS EAC4 bulk download via ADS API |
| `main.py` | AERONET EDA + AOD time-series plots |
| `visualize.py` | Interactive 3-hourly CAMS maps |
| `cart.py` | Dust transport route schematic (Cartopy) |
| `cjs_data.py` | In-situ aerosol optical property summary (SAE/AAE) |
