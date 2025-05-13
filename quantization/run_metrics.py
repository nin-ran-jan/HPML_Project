import argparse
import torch
import time
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import wandb


# Load llama model with optional quantization
def load_model(model_id, quant_mode):
    if quant_mode == "8bit":
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_8bit=True
        )
    elif quant_mode == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config
        )
    else:  # Default: float16
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )


# Evaluate perplexity on WikiText2 using sliding window
def evaluate_perplexity(model, tokenizer, stride=512, max_length=2048):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    nlls = []
    total_tokens = 0

    context_limit = min(max_length, model.config.max_position_embeddings)

    for i in tqdm(range(0, input_ids.size(1), stride)):
        begin_loc = max(i + stride - context_limit, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i

        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)

        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        total_tokens += trg_len

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()

    print(f"\n--- Perplexity ---")
    print(f"Perplexity: {ppl:.3f}")

    return {"perplexity/perplexity": ppl}



# Evaluate generation latency + throughput on WikiText2
def evaluate_generation(model, tokenizer, num_samples=200, max_prompt_tokens=128, max_new_tokens=128):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    prompts = []
    for entry in dataset:
        text = entry["text"].strip().replace("\n", " ")
        if len(tokenizer(text).input_ids) >= max_prompt_tokens:
            prompts.append(text)
        if len(prompts) >= num_samples:
            break

    total_tokens = 0
    total_time = 0.0

    print(f"\nRunning generation on {len(prompts)} prompts...")

    for raw_text in tqdm(prompts):
        tokenized = tokenizer(
            raw_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens
        ).to(model.device)

        with torch.no_grad():
            start = time.time()
            output = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for benchmarking
                pad_token_id=tokenizer.eos_token_id
            )
            end = time.time()

        new_tokens = output.shape[1] - tokenized["input_ids"].shape[1]
        total_tokens += new_tokens
        total_time += (end - start)

    latency = total_time / total_tokens
    throughput = total_tokens / total_time

    print(f"\n--- Generation Metrics ---")
    print(f"Latency per token: {latency * 1000:.3f} ms")
    print(f"Throughput: {throughput:.3f} tokens/sec")

    return {
        "generation/latency_per_token": latency,
        "generation/throughput": throughput
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaMA on WikiText2 for perplexity and generation speed")

    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID to load")
    parser.add_argument("--eval_mode", type=str, choices=["perplexity", "generation", "both"], default="both")
    parser.add_argument("--quant_mode", type=str, choices=["16bit", "8bit", "4bit"], default="16bit")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_prompt_tokens", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()

    model = load_model(args.model_id, args.quant_mode)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model.eval()

    run_name = f"{args.model_id.split('/')[-1]}-{args.quant_mode}-{args.eval_mode}"

    wandb.init(
        project="final_project",
        entity="ns3888-hpml",
        name=run_name,
        config=vars(args)
    )

    if args.eval_mode in ["perplexity", "both"]:
        metrics = evaluate_perplexity(model, tokenizer, args.stride, args.max_length)
        wandb.log(metrics)
    if args.eval_mode in ["generation", "both"]:
        metrics = evaluate_generation(
            model, tokenizer,
            num_samples=args.num_samples,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens
        )
        wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    main()