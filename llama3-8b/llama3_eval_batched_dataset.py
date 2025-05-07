import os, time, math, torch, pynvml
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
# import wandb

TARGET_ID = "meta-llama/Llama-3.1-8B"
DATASET = "wikitext"
CONF_NAME = "wikitext-2-raw-v1"
MAX_PROMPT = 128
GEN_TOKENS = 64
NUM_SAMPLES = 100
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_stats():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    return mem.used / 1e9, util.gpu


def main():

    print("Loading Dataset...")
    ds = load_dataset(DATASET, CONF_NAME, split="test")
    raw_texts = [ex["text"].strip().replace("\n", " ") for ex in ds.select(range(NUM_SAMPLES))]
    prompts = [" ".join(text.split()[:MAX_PROMPT]) for text in raw_texts if len(text.split()) >= 5]

    print("Loading tokenizer and model pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    pipe = pipeline(
        "text-generation",
        model=TARGET_ID,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.eos_token_id,
        device=0 if DEVICE == "cuda" else -1,
    )

    full_lat = full_thr = 0.0
    print("Starting batched evaluation...")

    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch = prompts[i:i + BATCH_SIZE]

        mem0, util0 = gpu_stats()
        torch.cuda.synchronize()
        t0 = time.time()

        outputs = pipe(
            batch,
            max_new_tokens=GEN_TOKENS,
            do_sample=True,
            return_full_text=False,
        )

        torch.cuda.synchronize()
        dt = time.time() - t0
        mem1, util1 = gpu_stats()

        batch_size = len(batch)
        lat = dt / (GEN_TOKENS * batch_size)
        thr = (GEN_TOKENS * batch_size) / dt

        full_lat += lat * batch_size
        full_thr += thr * batch_size


    n = len(prompts)
    print(
        f"\nResults\n"
        f"Throughput  : {full_thr / n:.2f} tok/s\n"
        f"Latency/tok : {full_lat / n:.4f} s\n"
    )


if __name__ == "__main__":
    main()
