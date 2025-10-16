#!/bin/bash
#
echo "eager mode"
ENABLE_EXPERIMENTAL_FLAGS=1 FORCE_EAGER=1 python test_fsdpa_perf_v0.py --batch 16 --q_len 1024 --kv_len 1024
ENABLE_EXPERIMENTAL_FLAGS=1 FORCE_EAGER=1 python test_fsdpa_perf_v0.py --batch 8 --q_len 2048 --kv_len 2048
ENABLE_EXPERIMENTAL_FLAGS=1 FORCE_EAGER=1 python test_fsdpa_perf_v0.py --batch 4 --q_len 4096 --kv_len 4096
ENABLE_EXPERIMENTAL_FLAGS=1 FORCE_EAGER=1 python test_fsdpa_perf_v0.py --batch 1 --q_len 16384 --kv_len 16384

ENABLE_EXPERIMENTAL_FLAGS=1 FORCE_EAGER=1 python test_fsdpa_perf_v0.py --batch 8 --q_len 256 --kv_len 1024
ENABLE_EXPERIMENTAL_FLAGS=1 FORCE_EAGER=1 python test_fsdpa_perf_v0.py --batch 8 --q_len 16 --kv_len 1024

echo "lazy mode"
python test_fsdpa_perf_v0.py --batch 16 --q_len 1024 --kv_len 1024
python test_fsdpa_perf_v0.py --batch 8 --q_len 2048 --kv_len 2048
python test_fsdpa_perf_v0.py --batch 4 --q_len 4096 --kv_len 4096
python test_fsdpa_perf_v0.py --batch 1 --q_len 16384 --kv_len 16384

python test_fsdpa_perf_v0.py --batch 8 --q_len 256 --kv_len 1024
python test_fsdpa_perf_v0.py --batch 8 --q_len 16 --kv_len 1024
