import math, time, torch, wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME        = "mistralai/Mistral-7B-v0.1"   
WANDB_PROJECT     = "final_project"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS    = 50
MAX_TEST_SAMPLES  = 100

wandb.init(project=WANDB_PROJECT,
           name=f"{MODEL_NAME.split('/')[-1]}-baseline-eval",
           config=dict(device=DEVICE, new_tokens=MAX_NEW_TOKENS))

tok   = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

ds      = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
samples = [s["text"] for s in ds.select(range(MAX_TEST_SAMPLES)) if s["text"].strip()]

lat_sum = thr_sum = 0.0
for text in samples:
    inp = tok(text, return_tensors="pt").to(DEVICE)
    t0  = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, use_cache=False)
    dt   = time.time() - t0
    gen  = out.shape[-1] - inp["input_ids"].shape[-1]
    lat_sum += dt / gen
    thr_sum += gen / dt

wandb.log(dict(avg_latency=lat_sum/len(samples),
               avg_throughput=thr_sum/len(samples)))

full_text = "\n\n".join(samples)
enc       = tok(full_text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    loss = model(**enc, labels=enc["input_ids"]).loss
ppl = math.exp(loss.item())

wandb.log(dict(perplexity=ppl))
print(f"{MODEL_NAME} baseline perplexity: {ppl:.2f}")
wandb.finish()
