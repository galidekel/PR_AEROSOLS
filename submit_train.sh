#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE

PYTHON=/home/projects/aihubadm/galidek/.conda/envs/geo-ds-env/bin/python3
CODE_DIR=/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS
DATA_SRC=/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS_DATA
MEMMAP=/home/labs/rudich/Rudich_Collaboration/PR_AEROSOLS_DATA/memmap

cd $CODE_DIR

if [ ! -f $MEMMAP/route.dat ]; then
    echo "[memmap] not found, running preprocessing..."
    mkdir -p $MEMMAP
    $PYTHON -u preprocess_memmap.py --config config_hpc.yaml --scratch $MEMMAP
    echo "[memmap] preprocessing done."
else
    echo "[memmap] found, skipping preprocessing."
fi

echo "[scratch] starting training..."
$PYTHON -u train_main.py --config config_hpc.yaml
