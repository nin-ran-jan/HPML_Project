#!/bin/bash

python quantization/run_metrics.py \
  --model_id meta-llama/Llama-3.1-8B \
  --quant_mode 4bit \
  --eval_mode both \
  --stride 512 \
  --max_length 2048 \
  --num_samples 200 \
  --max_prompt_tokens 64 \
  --max_new_tokens 128