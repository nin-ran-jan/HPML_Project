import os, time, math, argparse, torch, wandb, pynvml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET_ID = "meta-llama/Llama-2-70b-hf"
DRAFT_ID = "meta-llama/Llama-2-7b-hf"
DATASET = "wikitext"
CONF_NAME = "wikitext-2-raw-v1"
MAX_PROMPT = 128          # prompt truncation
GEN_TOKENS = 64           # tokens to generate
NUM_SAMPLES = 100
NUM_DRAFT = 4             # draft tokens per step
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_stats():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    return mem.used / 1e9, util.gpu  # GB, %


def calc_perplexity(model, tok, texts):
    enc = tok("\n\n".join(texts), return_tensors="pt").to(model.device)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return math.exp(loss.item())


def main():
    wandb.init(
        project="final_project",
        name="llama13B-7B_specdecode",
        config=dict(
            target=TARGET_ID,
            draft=DRAFT_ID,
            dataset=DATASET,
            num_draft_tokens=NUM_DRAFT,
            gen_tokens=GEN_TOKENS,
            samples=NUM_SAMPLES,
            prompt_len=MAX_PROMPT,
        ),
    )

    tok = AutoTokenizer.from_pretrained(TARGET_ID, use_fast=False)

    print("Loading target (13B)...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    print("Loading draft (7B)...")
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    ds = load_dataset(DATASET, CONF_NAME, split="test")
    texts = [ex["text"] for ex in ds.select(range(NUM_SAMPLES)) if ex["text"].strip()]

    full_lat = full_thr = full_accept = full_rb = 0.0

    for i, txt in enumerate(texts):
        prompt_ids = tok(
            txt, truncation=True, max_length=MAX_PROMPT, return_tensors="pt"
        ).to(DEVICE)

        mem0, util0 = gpu_stats()
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            out = target.generate(
                **prompt_ids,
                max_new_tokens=GEN_TOKENS,
                do_sample=False,
                draft_model=draft,
                num_draft_tokens=NUM_DRAFT,
                return_dict_in_generate=True,
                output_scores=False,
            )

        torch.cuda.synchronize()
        dt = time.time() - t0
        mem1, util1 = gpu_stats()

        gen = GEN_TOKENS
        lat = dt / gen
        thr = gen / dt

        spec = getattr(out, "speculative_details", None)
        if spec is not None:
            accept_ratio = spec["acceptance_mask"].float().mean().item()
            rollbacks = spec["num_rollbacks"]
        else:
            accept_ratio = float("nan")
            rollbacks = 0

        full_lat += lat
        full_thr += thr
        full_accept += accept_ratio
        full_rb += rollbacks

        if i < 10:
            wandb.log(
                dict(
                    sample=i,
                    latency_per_token=lat,
                    throughput=thr,
                    accept_ratio=accept_ratio,
                    rollbacks=rollbacks,
                    gpu_util=util1,
                    gpu_mem_GB=mem1,
                )
            )

    ppl = calc_perplexity(target, tok, texts)

    n = len(texts)
    wandb.log(
        dict(
            avg_latency_per_token=full_lat / n,
            avg_throughput=full_thr / n,
            avg_accept_ratio=full_accept / n,
            avg_rollbacks=full_rb / n,
            test_perplexity=ppl,
        )
    )

    print(
        f"\nResults\n"
        f"Throughput  : {full_thr / n:.2f} tok/s\n"
        f"Latency/tok : {full_lat / n:.4f} s\n"
        f"Accept-rate : {full_accept / n:.2%}\n"
        f"Rollbacks   : {full_rb / n:.2f} per seq\n"
        f"Perplexity  : {ppl:.2f}"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
