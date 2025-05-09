#!/bin/bash

python spec-decoding/llama3_specdecode.py \
  meta-llama/Llama-3.1-8B \
  --aux-model meta-llama/Llama-3.2-3B \
  --dtype bf16 \
  --target-quant 8bit \
  --draft-quant 4bit \
  --num-samples 500 \
  --assist-toks 3 \
  --gen-toks 128 \
  --wandb-project final_project \
  --wandb-entity ns3888-hpml \
  --wandb-run temp \
  --compile