# HPML Project: Optimizing LLM Inference with Speculative Decoding and Quantization

## Team Information
- **Team Name**: Pipe
- **Members**:
  - Niranjan Sundararajan (ns3888)
  - Michael Khanzadeh (mmk2258)
  - Ryan Huang (rh3129)

---

## 1. Problem Statement

The growing size of Large Language Models (LLMs) poses significant runtime and memory challenges, especially in resource-constrained environments. This project explores two complementary techniques to address these issues:
- **Speculative Decoding**: Accelerating autoregressive generation by having a smaller draft model suggest multiple tokens, which the larger target model then validates.
- **Quantization**: Reducing model precision (e.g., 8-bit, 4-bit) to improve latency and reduce memory footprint.

We benchmark these techniques using the LLaMA 3 model family and evaluate performance across metrics such as latency, throughput, perplexity, and code generation correctness (pass@1).

---

## 2. Model Description

We use open-source decoder-only LLMs from the **LLaMA 3** family:
- **Target Models**: LLaMA 3 8B
- **Draft Models**: LLaMA 3 3B and 1B

**Framework**: PyTorch (via HuggingFace Transformers)

**Quantization**: 
- 8-bit and 4-bit quantization using `bitsandbytes` for inference

**Custom Features**:
- Draft model adapter for speculative decoding using HuggingFace's `AssistedCandidateGenerator`
- Quantized evaluation pipeline for perplexity, throughput, and HumanEval

---

## 3. Final Results Summary

### LLAMA-3 8B/1B – Speculative Decoding (τ = 0.0001, No Sampling)

| Tok | Throughput (t/s) | Latency (ms/t) | Rollbacks | Acceptance | Speedup | GPU (%) |
|-----|------------------|----------------|-----------|------------|---------|---------|
| 3   | 19.995           | 50.013         | 2502      | 0.547      | 1.301   | 89.08   |
| 5   | 18.653           | 53.611         | 2700      | 0.440      | 1.214   | 86.79   |
| 7   | 16.696           | 59.896         | 2796      | 0.363      | 1.086   | 72.30   |

---

### LLAMA-3 8B/1B – Speculative Decoding (τ = 0.0001, With Sampling)

| Tok | Throughput (t/s) | Latency (ms/t) | Rollbacks | Acceptance | Speedup | GPU (%) |
|-----|------------------|----------------|-----------|------------|---------|---------|
| 3   | 19.833           | 50.420         | 2491      | 0.549      | 1.290   | 89.88   |
| 5   | 18.226           | 54.866         | 2785      | 0.431      | 1.186   | 76.08   |
| 7   | 16.510           | 60.570         | 2863      | 0.361      | 1.074   | 73.84   |

---

### LLAMA-3 8B/1B – Speculative Decoding (τ = 0.2, No Sampling)

| Tok | Throughput (t/s) | Latency (ms/t) | Rollbacks | Acceptance | Speedup | GPU (%) |
|-----|------------------|----------------|-----------|------------|---------|---------|
| 3   | 20.769           | 48.150         | 2428      | 0.609      | 1.351   | 81.27   |
| 5   | 20.535           | 48.697         | 2606      | 0.544      | 1.336   | 76.91   |
| 7   | 20.334           | 49.178         | 2653      | 0.500      | 1.323   | 77.06   |

---

### LLAMA-3 8B/1B – Quantized Draft/Target (Tok = 3, τ = 0.2)

| Config               | Throughput | Latency | Rollbacks | Acceptance | Speedup | GPU (%) |
|----------------------|------------|---------|-----------|------------|---------|---------|
| 8-bit Draft          | 12.818     | 78.018  | 2491      | 0.602      | 0.834   | 50.26   |
| 4-bit Draft          | 14.423     | 69.335  | 2883      | 0.539      | 0.938   | 67.08   |
| 8-bit Target         | 14.661     | 68.207  | 2491      | 0.601      | 0.954   | 46.25   |
| 8-bit T, 4-bit D     | 11.025     | 90.702  | 2900      | 0.537      | 0.717   | 41.54   |

---

### LLAMA-3 8B/3B – Speculative Decoding (τ = 0.2, No Sampling)

| Tok | Throughput | Latency | Rollbacks | Acceptance | Speedup | GPU (%) |
|-----|------------|---------|-----------|------------|---------|---------|
| 3   | 17.099     | 58.481  | 1967      | 0.673      | 1.112   | 88.14   |
| 5   | 16.580     | 60.313  | 2116      | 0.610      | 1.079   | 92.56   |
| 7   | 15.928     | 62.782  | 2194      | 0.559      | 1.036   | 92.61   |

---

### HumanEval Accuracy – Non-Sampled (LLAMA-3 8B)

| Quantization | Pass@1 |
|--------------|--------|
| 4-bit        | 29.2%  |
| 8-bit        | 32.3%  |
| 16-bit       | 33.6%  |

---

### WikiText-2 Generation – Throughput, Perplexity, Utilization

| Quantization | Throughput (t/s) | Latency (ms/t) | Perplexity | GPU (%) |
|--------------|------------------|----------------|------------|---------|
| 16-bit       | 15.499           | 64.519         | 5.563      | 97.97   |
| 8-bit        | 9.028            | 110.765        | 5.640      | 48.93   |
| 4-bit        | 21.036           | 47.538         | 6.242      | 70.37   |

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

As a dependency for the HumanEval experiments, clone the following harness repository:

```bash
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -e .
cd ..
```

---

### B. Wandb Dashboard

View training and evaluation metrics here: https://wandb.ai/ns3888-hpml/final_project/

---

### C. Inference

This repository focuses on **inference-only benchmarking**. No training is performed.

To run speculative decoding:
```bash
bash spec-decoding/run_specdecode.sh
```

---

### D. Evaluation


To evaluate the quantized model on HumanEval:
```bash
bash quantization/run_human_eval.sh
```

To compute perplexity and generation latency:
```bash
bash quantization/run_metrics.sh
```

---

### E. Quickstart: Minimum Reproducible Result

```bash
# Step 1: Clone and set up environment
git clone https://github.com/nin-ran-jan/HPML_Project.git
cd HPML_Project
pip install -r requirements.txt

# Step 2: Clone HumanEval for evaluation
git clone https://github.com/openai/human-eval.git
cd human-eval
pip install -e .
cd ..

# Step 3: Run speculative decoding
bash spec-decoding/run_specdecode.sh

# Step 4: Run HumanEval benchmark
bash quantization/run_human_eval.sh

# Step 5: Evaluate perplexity and throughput
bash quantization/run_metrics.sh
```
---

### F. Hyperparameter Explanation 

#### Speculative Decoding

Below are the primary arguments you can configure when running `run_specdecode.py`:

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--model`        | HuggingFace model ID for the target (e.g., `meta-llama/Llama-3.1-8B`)       |
| `--aux-model`    | HuggingFace model ID for the draft model (e.g., `meta-llama/Llama-3.2-1B`)  |
| `--dtype`        | Precision for inference (`bf16`, `fp16`, or `float`)                        |
| `--target-quant` | Quantization for the target model (`none`, `8bit`, `4bit`)                  |
| `--draft-quant`  | Quantization for the draft model (`none`, `8bit`, `4bit`)                   |
| `--assist-toks`  | Number of assistant tokens predicted by the draft model per step           |
| `--gen-toks`     | Number of tokens to generate                                                |
| `--assistant-confidence-threshold`     | The logit confidence threshold to early reject assistant tokens                                   |
| `--num-samples`  | Number of prompts to evaluate                                               |
| `--do-sample`    | Enable sampling during generation (`True` for stochastic, default is greedy)|
| `--compile`      | Whether to apply `torch.compile()` for the target model                     |
| `--wandb-*`      | Various arguments for setting project name, entity, and run name in WandB   |

You can adjust these parameters directly in the `bash spec-decoding/run_specdecode.sh` script or via command-line when calling `run_specdecode.py`.

---

#### HumanEval Evaluation

Below are the primary arguments you can configure when running `run_human_eval.py`:

| Argument             | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `--model_id`         | HuggingFace model ID for the LLM (e.g., `meta-llama/Llama-3.1-8B`)          |
| `--quant`            | Quantization level for inference (`16`, `8`, or `4`)                        |
| `--num_tasks`        | Number of HumanEval tasks to run (maximum: 164)                             |
| `--num_samples`      | Number of completions to generate per task                                  |
| `--max_tokens`       | Maximum number of tokens to generate for each completion                    |
| `--output_prefix`    | Prefix for output `.jsonl` and result files (e.g., `samples`)               |
| `--wandb_project`    | Project name to use for logging with Weights & Biases                       |
| `--wandb_entity`     | W&B entity or team name for logging (e.g., `ns3888-hpml`)                   |
| `--wandb_run`        | Optional run name for logging (autogenerated if not provided)               |
| `--sampling`         | Enable sampling-based generation (`True` for sampling, default is greedy)   |

You can modify these parameters directly in `quantization/run_human_eval.sh` or supply them manually via command line. This script automatically logs `pass@1` results to your Weights & Biases dashboard and evaluates completions using OpenAI's HumanEval harness.

---

#### Metrics Evaluation

Below are the primary arguments you can configure when running `run_metrics.py`:

| Argument               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `--model_id`           | HuggingFace model ID for the LLM (e.g., `meta-llama/Llama-3.1-8B`)          |
| `--eval_mode`          | Evaluation mode: `perplexity`, `generation`, or `both`                      |
| `--quant_mode`         | Quantization mode: `16bit`, `8bit`, or `4bit`                               |
| `--stride`             | Step size for sliding window in perplexity evaluation                       |
| `--max_length`         | Maximum context length for perplexity computation                           |
| `--num_samples`        | Number of prompts to evaluate during generation                             |
| `--max_prompt_tokens`  | Maximum number of tokens in each prompt                                     |
| `--max_new_tokens`     | Number of tokens to generate from each prompt                               |

These parameters control both perplexity (evaluated via a sliding window over WikiText-2) and generation benchmarking (measuring latency and throughput). All results are logged to Weights & Biases.


---

## 5. Notes
- Code is organized into `spec-decoding/`, `quantization/`, and `legacy-code/`
- Model checkpoints are auto-downloaded via HuggingFace Hub
- The `legacy-code/` folder contains early experiments and scripts using Mistral 7B and LLaMA 2 7B models. These were part of the initial phase before transitioning to the LLaMA 3 family due to performance and compatibility considerations
- For issues or questions, contact: niranjan.s@columbia.edu, mmk2258@columbia.edu, rh3129@columbia.edu