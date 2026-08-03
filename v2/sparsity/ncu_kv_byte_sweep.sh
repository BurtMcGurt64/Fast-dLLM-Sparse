#!/usr/bin/env bash
# Measure REAL DRAM bytes read per forward at B=32 across context lengths, then
# fit the weight floor (intercept) vs KV traffic (slope). No theoretical byte
# formulas — every number comes from Nsight Compute hardware counters.
#
# Usage:   bash v2/sparsity/ncu_kv_byte_sweep.sh
# Requires ncu with GPU perf-counter access (see ERR_NVGPUCTRPERM note in the .py).
#
# ncu replays every kernel in the range once for DRAM counters, so each L takes a
# few minutes on a 7B forward. That's the price of measuring instead of computing.
set -euo pipefail

SCRIPT=v2/sparsity/measure_kv_bytes.py
B=32
LENS=(1024 2048 4096 8192)
METRICS=dram__bytes_read.sum,dram__bytes_write.sum
GPU=${CUDA_VISIBLE_DEVICES:-0}

run_ncu () {  # $1=context_len  $2=out.csv  $3...=extra args
  local L=$1; local OUT=$2; shift 2
  echo "=== ncu  L=$L  B=$B  -> $OUT ==="
  CUDA_VISIBLE_DEVICES=$GPU ncu \
      --nvtx --nvtx-include "measured_step/" \
      --metrics "$METRICS" \
      --target-processes all \
      --csv --log-file "$OUT" \
      python "$SCRIPT" --context_len "$L" --batch_size "$B" "$@"
}

# dense sweep across context lengths (isolates weight floor vs KV slope)
for L in "${LENS[@]}"; do
  run_ncu "$L" "kvbytes_dense_L${L}.csv"
done

# one sparse point at 8k for a dense-vs-sparse traffic comparison
run_ncu 8192 "kvbytes_sparse_L8192.csv" --sparse

echo
echo "########## MEASURED weight-floor vs KV-slope fit ##########"
python "$SCRIPT" --fit \
  1024 kvbytes_dense_L1024.csv \
  2048 kvbytes_dense_L2048.csv \
  4096 kvbytes_dense_L4096.csv \
  8192 kvbytes_dense_L8192.csv

echo
echo "Dense vs sparse DRAM read @ L=8192 (grep the CSVs):"
python - <<'PY'
import sys; sys.path.insert(0, "v2/sparsity")
from measure_kv_bytes import _sum_metric
d = _sum_metric("kvbytes_dense_L8192.csv")
s = _sum_metric("kvbytes_sparse_L8192.csv")
print(f"  dense  : {d/1e9:7.3f} GB")
print(f"  sparse : {s/1e9:7.3f} GB   -> traffic reduction {d/s:.2f}x")
PY
