import torch
import time
import torch.nn as nn

import habana_frameworks.torch.hpu as hpu

from habana_frameworks.torch.hpex.kernels import FusedSDPA


device = "hpu"

query = torch.rand(32768, 1, 32, 128, dtype=torch.bfloat16).to(device) #sdpa_args["query"]
key = torch.rand(32768, 1, 8, 128, dtype=torch.bfloat16).to(device) #sdpa_args["key"] 
value = torch.rand(32768, 1, 8, 128, dtype=torch.bfloat16).to(device) # sdpa_args["value"]


def m(query, key, value):
  q, k, v = [x.transpose(0, 1).transpose(1, 2) for x in [query, key, value]]
  causal = True
  scale = None
  attn_mask = None
  
  attention_dropout = 0.0
  use_fast_softmax = "None"
  use_fused_sdpa_with_recompute = True
  
  context_layer = FusedSDPA.apply(
            q, k, v, attn_mask, attention_dropout, causal, scale,
            use_fast_softmax, use_fused_sdpa_with_recompute
        )
  
  # [b, np, sq, hn] --> [sq, b, np, hn]
  context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
  return context_layer

print("Begin running")
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
