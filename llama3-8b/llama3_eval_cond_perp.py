import os, time, math, torch, pynvml
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import wandb

TARGET_ID = "meta-llama/Llama-3.1-8B"
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
    return mem.used / 1e9, util.gpu  # GB, %

def main():
    wandb.init(
        project="final_project",
        entity="ns3888-hpml",
        name="llama3_8b_plain_inference",
        config=dict(
            target=TARGET_ID,
            dataset=DATASET,
            gen_tokens=GEN_TOKENS,
            samples=NUM_SAMPLES,
            prompt_len=MAX_PROMPT,
        ),
    )

    print("Loading Dataset...")
    ds = load_dataset(DATASET, CONF_NAME, split="test")
    texts = [ex["text"] for ex in ds.select(range(NUM_SAMPLES)) if ex["text"].strip()]

    print("Initializing pipeline for plain inference...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID)
    pipe = pipeline(
        "text-generation",
        model=TARGET_ID,
        torch_dtype=torch.bfloat16,
        device=0 if DEVICE == "cuda" else -1,
        pad_token_id=tokenizer.eos_token_id 
    )

    print("Loading model for perplexity computation...")
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, torch_dtype=torch.bfloat16, device_map="auto").eval()

    full_lat = full_thr = full_ppl = 0.0

    print("Starting evaluation...")
    for i, txt in tqdm(enumerate(texts), total=len(texts)):
        prompt = txt.strip().replace("\n", " ")
        if len(prompt.split()) < 5:
            continue
        prompt = " ".join(prompt.split()[:MAX_PROMPT])

        encoded_prompt = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        torch.cuda.synchronize(); t0 = time.time()
        mem0, util0 = gpu_stats()

        output = pipe(
            prompt,
            max_new_tokens=GEN_TOKENS,
            do_sample=True,
            return_full_text=False,
        )

        torch.cuda.synchronize()
        dt = time.time() - t0
        mem1, util1 = gpu_stats()

        generated_text = output[0]["generated_text"]
        full_input = tokenizer(prompt + generated_text, return_tensors="pt").to(DEVICE)
        input_ids = full_input["input_ids"]

        labels = input_ids.clone()
        labels[0, :encoded_prompt["input_ids"].size(1)] = -100

        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels).loss
        ppl = math.exp(loss.item())

        lat = dt / GEN_TOKENS
        thr = GEN_TOKENS / dt

        full_lat += lat
        full_thr += thr
        full_ppl += ppl

        wandb.log(
            dict(
                sample=i,
                latency_per_token=lat,
                throughput=thr,
                gpu_util=util1,
                gpu_mem_GB=mem1,
                perplexity=ppl,
            )
        )

    avg_lat = full_lat / len(texts)
    avg_thr = full_thr / len(texts)
    avg_ppl = full_ppl / len(texts)

    print(
        f"\nResults\n"
        f"Throughput  : {avg_thr:.2f} tok/s\n"
        f"Latency/tok : {avg_lat:.4f} s\n"
        f"Perplexity  : {avg_ppl:.2f}\n"
    )

    wandb.log(
        dict(
            avg_latency_per_token=avg_lat,
            avg_throughput=avg_thr,
            avg_perplexity=avg_ppl,
        )
    )
    wandb.finish()

if __name__ == "__main__":
    main()
