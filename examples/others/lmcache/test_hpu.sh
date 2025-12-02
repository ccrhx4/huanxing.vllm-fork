#!/bin/bash

VLLM_SKIP_WARMUP=true PT_HPU_GPU_MIGRATION=1 python disagg_prefill_lmcache_v0_prefill_only.py 
