#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE

PYTHON=/home/projects/aihubadm/galidek/.conda/envs/geo-ds-env/bin/python3
CODE_DIR=/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS
DATA_SRC=/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS_DATA
SCRATCH=/scratch/galidek_pr

cd $CODE_DIR

if [ ! -f $SCRATCH/route.dat ]; then
    echo "[scratch] memmap not found, running preprocessing..."
    mkdir -p $SCRATCH
    $PYTHON -u preprocess_memmap.py --config config_hpc.yaml --scratch $SCRATCH
    echo "[scratch] preprocessing done."
else
    echo "[scratch] memmap found, skipping preprocessing."
fi

echo "[scratch] starting training..."
$PYTHON -u train_main.py --config config_hpc.yaml
