---
library_name: peft
license: llama2
base_model: meta-llama/Llama-2-7b-hf
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: llama2-7b-baseline-wikitext
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: wikitext wikitext-2-raw-v1
      type: wikitext
      args: wikitext-2-raw-v1
    metrics:
    - type: accuracy
      value: 0.5598350909929907
      name: Accuracy
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama2-7b-baseline-wikitext

This model is a fine-tuned version of [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) on the wikitext wikitext-2-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.0909
- Accuracy: 0.5598

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- training_steps: 100

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.52.0.dev0
- Pytorch 2.6.0+cu124
- Datasets 3.5.0
- Tokenizers 0.21.1