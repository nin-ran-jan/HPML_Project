#!/bin/bash

python spec-decoding/llama3_specdecode.py \
  meta-llama/Llama-3.1-8B \
  --aux-model meta-llama/Llama-3.2-1B \
  --dtype bf16 \
  --target-quant none \
  --draft-quant none \
  --num-samples 500 \
  --assist-toks 3 \
  --gen-toks 128 \
  --assistant-confidence-threshold 0.2 \
  --wandb-project final_project \
  --wandb-entity ns3888-hpml \
  --wandb-run temp \
  --compile