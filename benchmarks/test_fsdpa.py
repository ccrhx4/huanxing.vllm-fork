import os
import torch
import time
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function

import habana_frameworks.torch.hpu as hpu

from habana_frameworks.torch.hpex.kernels import FusedSDPA

def is_profile_true():
    val = os.getenv("PROFILE")
    if val is None:
        return False
    # Normalize case
    val_lower = val.strip().lower()
    return val_lower in ("true", "1", "yes", "on")

device = "hpu"

batch = 8
q_len = 8
kv_len = 256  # past + maybe current
n_heads_q = 32
n_heads_kv = 8
head_dim = 128

# Dummy Q, K, V
# Create query with full heads
query = torch.randn(batch, n_heads_q, q_len, head_dim, device=device)

# Create key/value with fewer heads
key   = torch.randn(batch, n_heads_kv, kv_len, head_dim, device=device)
value = torch.randn(batch, n_heads_kv, kv_len, head_dim, device=device)


def m(query, key, value):
  causal = True
  scale = None
  attn_mask = None
  
  attention_dropout = 0.0
  use_fast_softmax = "None"
  use_fused_sdpa_with_recompute = True
  
  context_layer = FusedSDPA.apply(
            query, key, value, attn_mask, attention_dropout, causal, scale,
            use_fast_softmax, use_fused_sdpa_with_recompute
        )
  
  return context_layer

print("Begin running")
start = time.time()

if is_profile_true():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.HPU], with_stack=True) as prof:
        output = m(query, key, value)
    torch.hpu.synchronize()
    prof.export_chrome_trace("fsdpa_profile.json")
else:
    output = m(query, key, value)
    torch.hpu.synchronize()

duration = time.time() - start
print("Duration:", duration)


# print("OUTPUT", output)
