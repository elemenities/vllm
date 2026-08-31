#!/usr/bin/env bash
# Run only after the FP8 profile passes all startup and greedy-output checks.
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

exec python -m vllm.entrypoints.openai.api_server \
  --model sakamakismile/Qwen3.8-27B-MTP-NVFP4 \
  --served-model-name Qwen3.8-27B-MTP-NVFP4 \
  --host 127.0.0.1 \
  --port 8082 \
  --tensor-parallel-size 2 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype fp8 \
  --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --enforce-eager \
  --speculative-config '{"method":"dflash","model":"YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
