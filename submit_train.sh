#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE

PYTHON=/home/projects/aihubadm/galidek/.conda/envs/geo-ds-env/bin/python3
CODE_DIR=/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS
DATA_SRC=/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS_DATA
SCRATCH=/scratch/galidek_pr

cd $CODE_DIR

echo "[scratch] copying ERA5..."
mkdir -p $SCRATCH
cp -r $DATA_SRC/era5_dataset $SCRATCH/
echo "[scratch] copying CAMS..."
cp -r $DATA_SRC/cams_dataset $SCRATCH/
echo "[scratch] done, starting training..."

$PYTHON train_main.py --config config_hpc.yaml
