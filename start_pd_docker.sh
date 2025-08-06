#!/bin/bash

image=vllm_pd:ww31

VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

models=/mnt/disk02/hf_models/
workspace=/mnt/disk01/huanxing/vllm_pd/workdir
models_mnt_point=/dataset
workspace_mnt_point=/workspace/workdir

docker run \
  -it \
  --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  -e no_proxy=10.112.0.0/16,localhost,127.0.0.1,0.0.0.0 \
  -e VLLM_HOST_IP="$VLLM_HOST_IP" \
  -e GLOO_SOCKET_IFNAME=enp3s0f1 \
  -v /mnt/disk02/:/mnt/disk2 \
  --device=/dev:/dev \
  -v /dev:/dev \
  --cap-add=sys_nice \
  --cap-add SYS_PTRACE \
  --cap-add=CAP_IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --net=host \
  --ipc=host \
  -v "$models":"$models_mnt_point" \
  -v "$workspace":"$workspace_mnt_point" \
  "$image"

