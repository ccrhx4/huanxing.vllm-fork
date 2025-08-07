#!/bin/bash

model=/mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/
serverport=8868
serverip=localhost

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s 0.0.0.0:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

launch_proxy() {
  local n_repeat=$1
  source pd_xpyd/xpyd_start_proxy_dyn.sh 1 2 1 false benchmark $n_repeat &
  sleep 1
}

kill_proxy_server() {
  local port=$1
  lsof -t -i:$port | xargs -r kill -9
  sleep 1
}

benchmark() {
  local inputlen=$1
  local outputlen=$2


python3 benchmarks/benchmark_serving.py \
  --backend vllm \
  --model $model \
  --dataset-name sonnet \
  --request-rate inf \
  --host $serverip \
  --port $serverport \
  --sonnet-input-len $inputlen \
  --sonnet-output-len $outputlen \
  --sonnet-prefix-len 100 \
  --trust-remote-code \
  --max-concurrency 256 \
  --num-prompts 1 \
  --ignore-eos \
  --burstiness 1000 \
  --dataset-path benchmarks/sonnet.txt \
  --save-result

}

repeat=127

kill_proxy_server $serverport
launch_proxy $repeat
wait_for_server $serverport
benchmark 2000 2000
kill_proxy_server $serverport

launch_proxy $repeat
wait_for_server $serverport
benchmark 2000 2000
kill_proxy_server $serverport

launch_proxy $repeat
wait_for_server $serverport
benchmark 2000 2000
kill_proxy_server $serverport

repeat=63

echo "running $repeat======================="
kill_proxy_server $serverport
launch_proxy $repeat
wait_for_server $serverport
benchmark 2000 2000
kill_proxy_server $serverport

launch_proxy $repeat
wait_for_server $serverport
benchmark 2000 2000
kill_proxy_server $serverport

launch_proxy $repeat
wait_for_server $serverport
benchmark 2000 2000
