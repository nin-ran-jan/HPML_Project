# HPML_Project

This repository presents our work in benchmarking state-of-the-art LLM performance improvements through techniques such as **Speculative Decoding**, **Quantization**, and **Generation Evaluation**. The project was developed as part of the High Performance Machine Learning course offered by Prof. Kaoutar El Maghraoui at Columbia University.

---

## Table of Contents
- [Project Motivation](#project-motivation)
- [Hardware Setup](#hardware-setup)
- [Setup and Installation](#setup-and-installation)
- [Running Experiments](#running-experiments)
  - [Speculative Decoding](#speculative-decoding)
  - [Quantized Model Evaluation](#quantized-model-evaluation)
    - [HumanEval Benchmark](#humaneval-benchmark)
    - [Performance Metrics](#performance-metrics)
- [Logging and Monitoring](#logging-and-monitoring)
- [Observations and Findings](#observations-and-findings)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Project Motivation

In the initial stages, we worked with Mistral 7B and LLaMA 2 7B models. During speculative decoding experiments, we realized the draft models needed to be quantized due to memory constraints. However, quantization degraded performance significantly.

We later transitioned to the **LLaMA 3** family—particularly the 8B, 3B, and 1B models—which are more compatible with our runtime and provide better support for mixed-precision inference. Our final code uses these models. Earlier runs are archived in the `legacy-code/` directory.

---

## Hardware Setup

All experiments were conducted on **Google Cloud** instances with:
- NVIDIA L4 GPUs (24 GB VRAM)
- 16 vCPUs
- 64 GB RAM

Identical environments were maintained across all runs for consistency.

---

## Setup and Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/nin-ran-jan/HPML_Project.git
cd HPML_Project
pip install -r requirements.txt
```

If using Conda:

```bash
conda create -n hpml_env python=3.10
conda activate hpml_env
pip install -r requirements.txt
```

Main requirements:
- `transformers`
- `datasets`
- `torch`
- `bitsandbytes`
- `wandb`
- `tqdm`

---

# Running Experiments

## Speculative Decoding

To benchmark speculative decoding:

```bash
bash spec-decoding/run_specdecode.sh
```

This script evaluates speedups using a target model and a smaller draft model.

#### Key Arguments
- `--model`, `--aux-model`: Target and assistant model names
- `--dtype`: Data type (e.g., `bf16`, `fp16`)
- `--target-quant`, `--draft-quant`: Quantization level (`none`, `8bit`, `4bit`)
- `--assist-toks`: Number of assistant tokens per generation step
- `--do-sample`: Enable sampling (stochastic decoding)
- `--compile`: Apply `torch.compile` for speedup
- `--wandb-*`: Weights & Biases logging options

#### Example

```bash
python run_specdecode.py meta-llama/Llama-3.1-8B \
  --aux-model meta-llama/Llama-3.2-1B \
  --target-quant none --draft-quant none \
  --dtype bf16 --assist-toks 5 \
  --gen-toks 128 --num-samples 500 \
  --compile --do-sample \
  --wandb-project final_project \
  --wandb-entity ns3888-hpml \
  --wandb-run llama3-specdecode8-1_sampling_true_toks5_L4
```

---

## Quantized Model Evaluation

This section benchmarks quantized and non-quantized models using two methods:

### HumanEval Benchmark

To evaluate the functional correctness of generated Python code:

```bash
bash quantization/run_human_eval.sh
```

#### Description
This script uses [OpenAI’s HumanEval](https://github.com/openai/human-eval) dataset to evaluate generated code completions for correctness (e.g., `pass@1`).

#### Custom Arguments
- `--model_id`: Model name (e.g., `meta-llama/Llama-3.1-8B`)
- `--quant`: Quantization level (`16`, `8`, `4`)
- `--num_tasks`: Number of tasks (default: 40, max: 164)
- `--num_samples`: Completions per task
- `--sampling`: Enable sampling (default: greedy)
- `--wandb-*`: Weights & Biases logging

#### Example

```bash
python quantization/run_human_eval.py \
  --model_id meta-llama/Llama-3.1-8B \
  --quant 8 \
  --num_tasks 164 \
  --num_samples 1 \
  --wandb_project final_project \
  --wandb_entity ns3888-hpml \
  --sampling
```

Outputs:
- `samples.jsonl`: Completions
- `samples.jsonl_results.jsonl`: Evaluation results

---

### Performance Metrics

To evaluate **perplexity**, **latency**, and **throughput** on text generation:

```bash
bash quantization/run_metrics.sh
```

#### Description
Evaluates model performance on `wikitext-2-raw-v1` using:
- Sliding window perplexity
- Deterministic text generation latency
- Tokens/sec throughput

#### Key Arguments
- `--quant_mode`: Quant level (`16bit`, `8bit`, `4bit`)
- `--eval_mode`: `perplexity`, `generation`, or `both`
- `--num_samples`: Number of generation prompts
- `--stride`, `--max_length`: For perplexity windowing

#### Example

```bash
python quantization/run_metrics.py \
  --model_id meta-llama/Llama-3.1-8B \
  --quant_mode 4bit \
  --eval_mode both \
  --num_samples 200
```

All metrics are logged to Weights & Biases under relevant tags:
- `perplexity/perplexity`
- `generation/latency_per_token`
- `generation/throughput`

---

## Logging and Monitoring

All experiments use [Weights & Biases](https://wandb.ai/) for experiment tracking. Set your own `--wandb-project` and `--wandb-entity`, or modify the defaults:

```bash
wandb_project="final_project"
wandb_entity="ns3888-hpml"
```

Example metrics tracked:
```json
{
  "generation/latency_per_token": 0.0432,
  "generation/throughput": 23.14,
  "humaneval/pass@1": 0.82,
  "perplexity/perplexity": 9.83
}
```

TODO: Add public WandB link here

---

## Observations and Findings

### Speculative Decoding

1. The 8B–1B model pair consistently gave the best results with 3 assistant tokens, no sampling, and a confidence threshold of 0.2.
2. The number of assistant tokens had a clear tradeoff: too few (ex. 1) slowed things down due to limited draft generation, while too many led to excessive rejections and rollbacks.
3. Greedy decoding (no sampling) consistently outperformed sampled generation in terms of speed and alignment between draft and target outputs.
4. Using a larger draft model like 3B improved acceptance rates slightly but resulted in slower generation overall—net performance dropped.
5. Across the board, quantized models showed lower GPU utilization and worse performance, likely due to the L4 GPU’s hardware bias toward bf16 rather than int8 or nf4.

### Quantized Model Evaluation

1. TODO: Add results table summarizing `pass@1` across quantization levels

---

## Results

TODO: Add benchmark tables for latency, throughput, perplexity, and pass@1.

---

## Acknowledgements

This work was done as part of the High Performance Machine Learning course at Columbia University. We thank the open-source contributors of HuggingFace, OpenAI's HumanEval, and BitsAndBytes for tooling support. We also thank Google Cloud for providing the development environment. 
