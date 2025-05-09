
# Storyline for Project and Motivation

In the initial phases of the experiments, we were using 2 sets of models -- mistral 7B and llama2 7B for our evaluation. When we started experimenting with speculative decoding, we came to the conclusion that the draft versions of these models had to be quantized for our runtime. However, the results with quantized models as draft models were not ideal at the time, so we decided to switch to the Llama3 family of models. Here, we primarily use 8B, 3B, and 1B family of models, all which are compatible with our compute environment. 

# Hardware used

We use L4 GPUs with 24 GB of VRAM for all of our experiments. Each machine has an identical setup with a total of 16 vCPUs and 64 GB of memory (RAM). 

# Observations from Results

## Spec Decoding
1. We observed that the speculative decoding for the 8B 1B model performs the best, with number of assistant tokens = 3, no sampling, and confidence threshold = 0.2. 
2. We observed a trend with number of assistant tokens and speedup. There was an optimal value at 3 -> 1 was too small for generation so draft model was bottlenecking and high values were producing too many rejected tokens, resulting many more rollbacks causing a net negative effect. 
3. Without sampling, the model was deterministic in selecting the outputs with the maximum logit value for prediction. Hence, these results were better than sampled results. 
4. As we increased the base model size from 1B to 3B, we saw a performance dip. Even though the model has a better acceptance rate of tokens, the larger size of it inhibits a fast output of multiple assistant tokens and causes a net slowdown in speed. 
5. With all combinations of quantization, the model shows a dip in performance. We can see that in all of the quantized settings the GPU utilization is lesser as compared to the other cases. We attribute this to our hardware that has specific optimizations for bf16, as compared to quantized nf4 and int8. 

## Human eval

1. TODO 


# HPML_Project

TODO: Benchmarking SoTA LLM performance improvements by introducing techniques like KV Caching, Speculative Decoding, and Quantization. This is a course project for the High Performance Machine Learning course offered by Prof. Kaoutar El Maghraoui at Columbia University.


---

## Table of Contents
- [Storyline and Project Motivation](#storyline-and-project-motivation)
- [Hardware Used](#hardware-used)
- [Setup and Installation](#setup-and-installation)
- [Running Experiments](#running-experiments)
  - [Speculative Decoding](#speculative-decoding)
  - [HumanEval](#human-eval-on-quantized-models)
- [Usage](#usage)
- [Logging and Evaluation](#logging-and-evaluation)
- [Observations from Results](#observations-from-results)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

---

## Storyline and Project Motivation

In the initial phases of the experiments, we were using two sets of models—Mistral 7B and LLaMA 2 7B—for our evaluation. When we began experimenting with speculative decoding, we realized that draft versions of these models needed to be quantized to fit our runtime constraints.

However, the results with quantized models as draft models were suboptimal, leading us to transition to the LLaMA 3 family. Here, we primarily use the 8B, 3B, and 1B models, all of which are compatible with our compute setup. These models allowed us to explore a broader range of configurations and improved both performance and deployment feasibility.

The code for all of our runs from before are in the ```legacy-code/``` folder. We have cited the important parts which were motivations for our final output code. 

---

## Hardware Used

All experiments are conducted on NVIDIA L4 GPUs with 24 GB of VRAM from Google Cloud Platform. Each compute instance is provisioned with:
- 16 vCPUs
- 64 GB system RAM
- Identical software and driver environments to ensure consistent benchmarking

---

## Setup and Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/nin-ran-jan/HPML_Project.git
cd HPML_Project
pip install -r requirements.txt
```

If you are using Conda:

```bash
conda create -n hpml_env python=3.10
conda activate hpml_env
pip install -r requirements.txt
```

Ensure the following libraries are installed:
- `torch`
- `transformers`
- `datasets`
- `bitsandbytes`
- `wandb`
- `tqdm`

---

## Running Experiments

### Speculative Decoding

To run speculative decoding, use the following shell script:

```bash
bash spec-decoding/run_specdecode.sh
```

#### Required Parameters

The script accepts the following arguments:

- `model`: The target LLM model. Example: `meta-llama/Llama-3.1-8B`
- `aux-model`: The assistant (draft) model used for speculative decoding. Example: `meta-llama/Llama-3.2-1B`
- `dtype`: Data type for model loading. Default: `bf16`
- `--target-quant`: Quantization setting for target model. Options: `none`, `8bit`, `4bit`
- `--draft-quant`: Quantization setting for draft model. Options: `none`, `8bit`, `4bit`
- `--num-samples`: Number of test samples. Default: `500`
- `--max-prompt`: Prompt length. Default: `128`
- `--gen-toks`: Number of tokens to generate. Default: `128`
- `--assist-toks`: Number of assistant tokens per step. Default: `5`
- `--compile`: Whether to use `torch.compile` for the models
- `--do-sample`: Enable sampling instead of greedy decoding
- `--wandb-project`: Weights & Biases project name
- `--wandb-entity`: Weights & Biases entity name
- `--wandb-run`: Weights & Biases run name

#### Example

```bash
python run_specdecode.py meta-llama/Llama-3.1-8B \
  --aux-model meta-llama/Llama-3.2-1B \
  --dtype bf16 \
  --target-quant none \
  --draft-quant none \
  --num-samples 500 \
  --assist-toks 5 \
  --gen-toks 128 \
  --compile \
  --do-sample \
  --wandb-project final_project \
  --wandb-entity ns3888-hpml \
  --wandb-run llama3-specdecode8-1_sampling_true_toks5_L4
```

---

## Human Eval on Quantized Models

To run human evals:

```bash
git clone https://github.com/openai/human-eval.git
pip install -e human-eval
```

Then use the provided evaluation scripts to assess code generation quality and correctness.

---

## Usage

- The main script loads both the target and draft models and benchmarks assisted (speculative) decoding performance.
- It uses `wikitext-2-raw-v1` as the test dataset, filtering out short samples.

---

## Logging and Evaluation

Evaluation metrics are automatically logged to Weights & Biases:
- Latency per token
- Throughput (tokens/sec)
- Accept rate (Number of tokens accepted by the draft model)
- Number of rollbacks (In case of mismatch between draft and target model, a rollback occurs. )

Sample wandb logs include:

```json
{
  "assisted_latency_tok": 0.04321,
  "assisted_thr": 23.14,
  "accept_rate": 0.82,
  "total_rollbacks": 47
}
```

TODO: provide wandb link to the project. 

---

## Observations from Results

### Speculative Decoding

1. The best performance was achieved with the 8B–1B pairing, using 3 assistant tokens, no sampling, and a confidence threshold of 0.2.
2. There is a non-linear relationship between assistant token count and speedup. A value of 3 strikes a balance; 1 token leads to slow draft generation, while higher values cause excessive rejections and rollbacks.
3. Greedy decoding (no sampling) produces more stable results with better performance compared to stochastic sampling.
4. Increasing the size of the draft model (e.g., from 1B to 3B) results in performance degradation. Despite higher acceptance rates, the slower draft generation due to model size leads to a net slowdown.
5. Across all quantization configurations, performance consistently drops. GPU utilization is noticeably lower with quantized models, likely due to hardware-level optimizations that favor `bf16` over `int8` or `nf4`.

### Human Eval

1. TODO

---

## Results

TODO: need to add tables from the experiments tracked

## Acknowledgements

This project is part of the HPML course at Columbia University. Models and datasets used are sourced from HuggingFace and OpenAI's HumanEval.

