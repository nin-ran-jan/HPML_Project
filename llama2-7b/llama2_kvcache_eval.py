import math, time, torch, wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_NEW_TOKENS = 50
MAX_TEST_SAMPLES = 100
USE_KV_CACHE = True


WANDB_PROJECT = "final_project"
MODEL_DIR = "llama2-7b-baseline-wikitext"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(
    project=WANDB_PROJECT,
    name=f"llama2-7b-eval-wikitext-{'kvcache' if USE_KV_CACHE else 'nocache'}",
    config={
        "eval_max_new_tokens": MAX_NEW_TOKENS,
        "kv_cache": USE_KV_CACHE,
        "device": DEVICE,
    },
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto",
                                             torch_dtype=torch.float16, local_files_only=True)
model.eval()
model.config.use_cache = USE_KV_CACHE

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
samples = [s["text"] for s in ds.select(range(MAX_TEST_SAMPLES)) if s["text"].strip()]

lat_sum, thr_sum = 0., 0.
for i, prompt in enumerate(samples):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, use_cache=USE_KV_CACHE)
    dt = time.time() - t0
    gen = out.shape[-1] - inputs["input_ids"].shape[-1]
    lat, thr = dt / gen, gen / dt
    lat_sum += lat; thr_sum += thr

    if i < 10:
        wandb.log({"sample_id": i, "latency_per_token": lat,
                   "throughput": thr,
                   "generation": tokenizer.decode(out[0], skip_special_tokens=True)})

torch.cuda.empty_cache()
model.config.use_cache = False

enc = tokenizer(" ".join(samples), return_tensors="pt").to(DEVICE)
with torch.no_grad():
    loss = model(**enc, labels=enc["input_ids"]).loss
perp = math.exp(loss.item())

wandb.log({"avg_latency_per_token": lat_sum/len(samples),
           "avg_throughput": thr_sum/len(samples),
           "test_perplexity": perp})
print(f"KV‑cache {USE_KV_CACHE} | Perplexity {perp:.2f}")

wandb.finish()
