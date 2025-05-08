#!/bin/bash

python spec-decoding/llama3/specdecode8-1/llama3_specdecode8-1_compare_against_baseline.py \
  meta-llama/Llama-3.1-8B \
  --aux-model meta-llama/Llama-3.2-1B \
  --dtype bf16 \
  --target-quant none \
  --draft-quant none \
  --num-samples 20 \
  --assist-toks 16 \
  --gen-toks 128 \
  --compile \
  --do-sample \
  --wandb-project final_project \
  --wandb-entity ns3888-hpml \
  --wandb-run llama3-specdecode8-1_sampling_true_toks20_L4