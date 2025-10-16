import argparse
import torch
import time
import torch.nn as nn

import habana_frameworks.torch.hpu as hpu

from habana_frameworks.torch.hpex.kernels import FusedSDPA


device = "hpu"

parser = argparse.ArgumentParser(description="GQA attention with variable batch & q_len")
parser.add_argument("--batch", type=int, required=True, help="Batch size")
parser.add_argument("--q_len", type=int, required=True, help="Query length")
parser.add_argument("--kv_len", type=int, default=None, help="Key/Value length (defaults to 2 * q_len if not provided)")

args = parser.parse_args()

batch = args.batch
q_len = args.q_len
kv_len = args.kv_len  # past + maybe current

# Llama
n_heads_q = 32
n_heads_kv = 8
head_dim = 128

# GLM config TP4
# n_heads_q = 24

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
  use_fast_softmax = "fast"
  use_fused_sdpa_with_recompute = True
  
  context_layer = FusedSDPA.apply(
            query, key, value, attn_mask, attention_dropout, causal, scale,
            use_fast_softmax, use_fused_sdpa_with_recompute
        )
  
  return context_layer

print("Begin running", batch, q_len, kv_len)
start = time.time()

output = m(query, key, value)
output = output.to("cpu")

duration = time.time() - start
print("Duration:", duration)


iterations = 100
print("Begin perf running for:", iterations)

start = time.time()
for i in range(iterations):
  output = m(query, key, value)
  output = output[0, 0, 0, 0]
  output = output.to("cpu")
duration = time.time() - start
print("Duration:", duration)

#print("OUTPUT", output)
