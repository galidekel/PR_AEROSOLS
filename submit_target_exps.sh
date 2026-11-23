#!/bin/bash
# Target / loss experiments on CV fold 3 (val = 2023-2024, the same window as the earlier runs).
# Baseline to compare with: outputs/run_3_stations_cv4_f3  (raw AOD500, MSE).
# Run from the repo directory on the HPC:   bash submit_target_exps.sh
# Submit only some of them by editing the list below.
PY=/home/projects/aihubadm/galidek/.conda/envs/geo-ds-env/bin/python3
FOLD=3
submit () {   # $1 = run suffix, $2 = config file, rest = extra train_main args
  name=$1; cfg=$2; shift 2
  bsub -q long-gpu -R "rusage[mem=320000]" -R "affinity[thread*8]" \
       -o ./outputs/train_out_$name.txt -e ./outputs/train_err_$name.txt \
       -gpu "num=1:j_exclusive=yes:gmem=40GB" \
       /bin/bash -c "export HDF5_USE_FILE_LOCKING=FALSE && $PY train_main.py --config $cfg --run_name run_3_stations_cv4_f${FOLD}_$name --val_fold $FOLD $*"
}
mkdir -p outputs
submit anom   config_hpc_cv4.yaml      --set target_anomaly=true select_metric=aod_mse           # item 6: anomaly target
submit log    config_hpc_cv4.yaml      --set target_transform=log select_metric=aod_mse          # item 7: log AOD
submit wgt    config_hpc_cv4.yaml      --set high_aod_weight=3 select_metric=aod_mse             # item 7: dust days count 4x
submit w1020  config_hpc_cv4_1020.yaml --set event_thr=0.2 select_metric=aod_mse                 # item 7: coarse-dominated 1020 nm target
