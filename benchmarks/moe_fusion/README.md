# MoE interleaved-fusion baseline-vs-fusion E2E benchmark

Orchestrates a baseline-vs-fusion `vllm bench serve` comparison for the
interleaved single-accumulator SwiGLU MoE fusion kernel
(`VLLM_XPU_FUSED_MOE_INTERLEAVED`, see `vllm_xpu_kernels.fused_moe_interface`).

Scripts:
- `start_server.sh` — starts a `vllm serve` OpenAI-compatible server with
  the flags used for this benchmark (XPU graph mode, block-size 64, etc).
- `bench.sh` — runs a `vllm bench serve` sweep across a list of
  concurrency levels against an already-running server.
- `run_moe_fusion_bench_compare.sh` — orchestrates both of the above for
  the `baseline` (fusion disabled) and `fusion` (fusion enabled)
  configurations, saving results under per-config subdirectories.

## Usage

```bash
# Default: Qwen3-30B-A3B, TP=4, concurrency 1-30, both configs.
benchmarks/moe_fusion/run_moe_fusion_bench_compare.sh both

# Only one configuration:
benchmarks/moe_fusion/run_moe_fusion_bench_compare.sh baseline
benchmarks/moe_fusion/run_moe_fusion_bench_compare.sh fusion

# Override model / result dir / concurrency levels:
MODEL=/hf_models/Qwen3.5-35B-A3B \
RESULT_ROOT=/work/bench_results_moe_interleaved_fusion_qwen35_35b_a3b \
CONCURRENCY_LIST="1 2 4 6 8 10 12 14 16 18 20 22" \
benchmarks/moe_fusion/run_moe_fusion_bench_compare.sh both
```

Environment overrides (all optional):
- `MODEL` — HF model path/id (default `/hf_models/Qwen3-30B-A3B`).
- `TP_SIZE` — tensor-parallel size (default `4`).
- `PORT` — server port (default `8115`).
- `RESULT_ROOT` — output directory root (default
  `/work/bench_results_moe_interleaved_fusion`); results land in
  `$RESULT_ROOT/{baseline,fusion}/`.
- `CONCURRENCY_LIST` — space-separated concurrency levels to sweep.
- `INPUT_LEN` / `OUTPUT_LEN` — random dataset input/output token lengths.

Results and interpretation for prior runs are documented under
`/work/fusemlp/E2E_BENCHMARK_RESULTS_*.md` (Qwen3-30B-A3B and
Qwen3.5-35B-A3B, TP=4).
