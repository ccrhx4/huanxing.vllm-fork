import argparse
import itertools
import math
import random
import time
from typing import List, Optional
from typing import (Any, AsyncIterator, Awaitable, Callable, Dict, Generic,
                    Hashable, List, Optional, OrderedDict, Set, Tuple, TypeVar,
                    Union)

import torch
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_kv_cache_torch_dtype)

if torch.cuda.is_available():
    from vllm_flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

NUM_BLOCKS = 2580
PARTITION_SIZE = 512


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> tuple[list[torch.tensor], list[torch.tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    key_cache_shape = (num_blocks, block_size, num_heads, head_size)
    key_caches: list[torch.tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            raise valueerror(
                f"does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, block_size, num_heads, head_size)
    value_caches: list[torch.tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=torch_dtype,
                                  device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            raise valueerror(
                f"does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches

@torch.inference_mode()
def main(
    version: str,
    num_seqs: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "hpu",
    kv_cache_dtype: Optional[str] = None,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)

    if device == "hpu":
        from vllm.utils import get_max_shared_memory_bytes, is_hip, is_hpu
        from vllm.hpu import ops, cache_ops
        import habana_frameworks.torch as ht
        from vllm.hpu.utils import Matmul, Softmax, VLLMKVCache

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif is_hpu():
        torch.hpu.manual_seed(seed)
    
    torch.set_default_device(device)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    query.uniform_(-scale, scale)
    
    key = torch.empty(num_seqs,
                        num_kv_heads,
                        head_size,
                        dtype=dtype,
                        device=device)

    key.uniform_(-scale, scale)
    
    value = torch.empty(num_seqs,
                        num_kv_heads,
                        head_size,
                        dtype=dtype,
                        device=device)

    value.uniform_(-scale, scale)


    print("query shape: ", query.shape)
    print("query shape: ", key.shape)
    print("query shape: ", value.shape)

    assert num_query_heads % num_kv_heads == 0
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device=device)

    seq_lens = [seq_len for _ in range(num_seqs)]
    print(seq_lens)

    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int, device=device)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: List[List[int]] = []
    block_indices = []
    block_offsets = []
    position = max_seq_len - 1

    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)
        block_indice = block_table[position // block_size]
        block_offset = position % block_size
        block_indices.append(block_indice)
        block_offsets.append(block_offset)

    #block_tables = torch.tensor(block_tables_lst,
    #                            dtype=torch.int,
    #                            device=device)

    blocks_used = [len(bt) for bt in block_tables_lst]
    block_list = list(itertools.chain(*block_tables_lst))
    block_list = torch.tensor(block_list, dtype=torch.int, device=device)
    
    block_mapping = [[i] * bu for i, bu in enumerate(blocks_used)]
    block_mapping = list(itertools.chain(*block_mapping))

    block_mapping = torch.tensor(block_mapping, dtype=torch.int, device=device)
    block_mapping = torch.nn.functional.one_hot(block_mapping, num_classes=num_seqs).to(dtype)

    block_indices = torch.tensor(block_indices, dtype=torch.int, device=device)
    block_offsets = torch.tensor(block_offsets, dtype=torch.int, device=device)

    # Create the KV cache.
    key_caches, value_caches = create_kv_caches_with_random(NUM_BLOCKS,
                                                        block_size,
                                                        1,
                                                        num_kv_heads,
                                                        head_size,
                                                        kv_cache_dtype,
                                                        dtype,
                                                        seed,
                                                        device=device)

    key_cache, value_cache = key_caches[0], value_caches[0]

    vllm_key_cache = VLLMKVCache()
    vllm_value_cache = VLLMKVCache()

    key_cache = vllm_key_cache(key, key_cache, block_indices, block_offsets)
    value_cache = vllm_value_cache(value, value_cache, block_indices, block_offsets)

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v2":
        num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        tmp_output = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
    
    # Using default kv_scale
    kv_scale = 1.0

    block_bias = torch.empty(block_list.numel(),
                        block_size,
                        dtype=dtype,
                        device=device)

    block_bias.uniform_(-scale, scale)
    print("attn bias shape: ", block_bias.shape)

    #prepare for HPU graph mode
    if not torch.cuda.is_available():
        hpu_stream = ht.hpu.Stream()
        cache = {}

    def page_attention_capture_replay(
            query,
            key_cache,
            value_cache,
            block_list,
            block_mapping,
            block_bias,
            scale,
            qk_matmul,
            kv_matmul,
            keys_fetch_func,
            values_fetch_func,
    ):
        inputs = [
            query,
            key_cache,
            value_cache,
            block_list,
            block_mapping,
            block_bias,
            scale,
            qk_matmul,
            kv_matmul,
            keys_fetch_func,
            values_fetch_func,
        ]

        h = ht.hpu.graphs.input_hash(inputs)
        cached = cache.get(h)

        if cached is None:
            with ht.hpu.stream(hpu_stream):
                graph = ht.hpu.HPUGraph()
                graph.capture_begin()
                
                outputs = ops.flat_pa(
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_list=block_list,
                    block_mapping=block_mapping,
                    block_bias=block_bias,
                    scale=scale,
                    qk_matmul_op=qk_matmul,
                    kv_matmul_op=kv_matmul,
                    keys_fetch_func=keys_fetch_func,
                    values_fetch_func=values_fetch_func,
                )

                graph.capture_end()
                graph_inputs = inputs
                graph_outputs = outputs
                cache[h] = ht.hpu.graphs.CachedParams(graph_inputs, graph_outputs, graph)
                
            return outputs
        
        ht.hpu.graphs.copy_to(cached.graph_inputs, inputs)
        cached.graph.replay()
        ht.core.hpu.default_stream().synchronize()

        return cached.graph_outputs

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            output = flash_attn_with_kvcache(
                query.unsqueeze(1),
                key_cache,
                value_cache,
                cache_seqlens=seq_lens,
                block_table=block_tables,
                softmax_scale=scale,
                causal=True,
                alibi_slopes=alibi_slopes,
            )
        
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        return (end_time - start_time) / num_iters
            
    def run_hpu_benchmark(num_iters: int, profile: bool = False) -> float:
        if is_hpu():
            torch.hpu.synchronize()

        start_time = time.perf_counter()

        for _ in range(num_iters):
            if is_hpu():
                output = page_attention_capture_replay(
                    query,
                    key_cache,
                    value_cache,
                    block_list,
                    block_mapping,
                    block_bias,
                    kv_scale,
                    Matmul(),
                    Matmul(),
                    vllm_key_cache.fetch_from_cache,
                    vllm_value_cache.fetch_from_cache,
                )
            else:
                raise ValueError(f"Invalid version: {version}")
        if is_hpu():
            torch.hpu.synchronize()

        end_time = time.perf_counter()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    
    if torch.cuda.is_available():
        run_benchmark = run_cuda_benchmark
    else:
        run_benchmark = run_hpu_benchmark

    latency = run_benchmark(num_iters=3, profile=False)
    print(f"Kernel warmup time: {latency * 1000000:.3f} us")

    # Benchmark.
    print("Start benchmarking...")
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["v1", "v2"],
                        default="v1")
    
    parser.add_argument("--device", type=str, default="hpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 192, 256],
                        default=128)
    parser.add_argument("--block-size", type=int, choices=[16, 32, 128], default=128)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="bfloat16")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        help="Data type for kv cache storage. If 'auto', will use model "
        "data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. "
        "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)")
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    main(
        version=args.version,
        num_seqs=args.batch_size,
        seq_len=args.seq_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        hidden_size=4096,
        head_size=args.head_size,
        block_size=args.block_size,
        use_alibi=args.use_alibi,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
        kv_cache_dtype=args.kv_cache_dtype,
        device=args.device,
    )
