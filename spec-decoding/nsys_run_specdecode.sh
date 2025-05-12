#!/bin/bash
nsys profile \
  --trace=cuda,nvtx,cudnn,osrt \
  --output=spec-decoding/llama3/nsys/llama3_specdecode8-1_sampling_false_toks3_0.2_L4 \
  --wait=all \
  --force-overwrite=true \
  bash spec-decoding/run_specdecode.sh