import os, time, math, torch, wandb, pynvml
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

TARGET_ID = "meta-llama/Llama-3.1-8B"
DRAFT_ID = "meta-llama/Llama-3.2-1B"
DATASET = "wikitext"
CONF_NAME = "wikitext-2-raw-v1"
MAX_PROMPT = 128
GEN_TOKENS = 64
NUM_SAMPLES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gpu_stats():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    return mem.used / 1e9, util.gpu 


def calc_perplexity(model, tokenizer, texts):
    inputs = tokenizer("\n\n".join(texts), return_tensors="pt").to(model.device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return math.exp(loss.item())


def main():
    # wandb.init(
    #     project="final_project",
    #     name="llama3_specdecode_pipeline",
    #     config=dict(
    #         target=TARGET_ID,
    #         draft=DRAFT_ID,
    #         dataset=DATASET,
    #         gen_tokens=GEN_TOKENS,
    #         samples=NUM_SAMPLES,
    #         prompt_len=MAX_PROMPT,
    #     ),
    # )

    print("Loading Dataset...")
    ds = load_dataset(DATASET, CONF_NAME, split="test")
    texts = [ex["text"] for ex in ds.select(range(NUM_SAMPLES)) if ex["text"].strip()]

    print("Initializing pipeline with speculative decoding...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    pipe = pipeline(
        "text-generation",
        model=TARGET_ID,
        assistant_model=DRAFT_ID,
        torch_dtype=torch.bfloat16,
        pad_token_id=tokenizer.eos_token_id 
    )

    full_lat = full_thr = 0.0

    print("Starting evaluation...")
    for i, txt in tqdm(enumerate(texts), total=len(texts)):
        prompt = txt.strip().replace("\n", " ")
        if len(prompt.split()) < 5:
            continue

        prompt = " ".join(prompt.split()[:MAX_PROMPT])

        mem0, util0 = gpu_stats()
        torch.cuda.synchronize()
        t0 = time.time()

        output = pipe(
            prompt,
            max_new_tokens=GEN_TOKENS,
            do_sample=True,
            return_full_text=False,
        )

        torch.cuda.synchronize()
        dt = time.time() - t0
        mem1, util1 = gpu_stats()

        lat = dt / GEN_TOKENS
        thr = GEN_TOKENS / dt

        full_lat += lat
        full_thr += thr

        # if i < 10:
        #     wandb.log(
        #         dict(
        #             sample=i,
        #             latency_per_token=lat,
        #             throughput=thr,
        #             gpu_util=util1,
        #             gpu_mem_GB=mem1,
        #         )
        #     )

    avg_lat = full_lat / len(texts)
    avg_thr = full_thr / len(texts)

    print(
        f"\nResults\n"
        f"Throughput  : {avg_thr:.2f} tok/s\n"
        f"Latency/tok : {avg_lat:.4f} s\n"
    )

    # around 24 tokens per second

    # wandb.log(
    #     dict(
    #         avg_latency_per_token=avg_lat,
    #         avg_throughput=avg_thr,
    #     )
    # )
    # wandb.finish()


if __name__ == "__main__":
    main()
