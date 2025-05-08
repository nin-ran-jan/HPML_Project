import math, time, torch, wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL  = "mistralai/Mistral-7B-v0.1"
ADAPTER_DIR = "./mistral-baseline-wikitext-quant-final"  
WANDB_PROJECT     = "final_project"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS    = 50
MAX_TEST_SAMPLES  = 100

wandb.init(
    project=WANDB_PROJECT,
    entity="ns3888-hpml",
    name="mistral-7b-ft-8bit-nokvcache",
    config=dict(quantized=True, kv_cache=False, fine_tuned=True,
                max_new_tokens=MAX_NEW_TOKENS, samples=MAX_TEST_SAMPLES,
                device=DEVICE)
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
base    = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            quantization_config=bnb_cfg)

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
model.config.use_cache = False       

print("Loaded adapter:", model.peft_config.keys())

ds      = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
samples = [s["text"] for s in ds.select(range(MAX_TEST_SAMPLES)) if s["text"].strip()]

lat_sum = thr_sum = 0.0
for i, text in enumerate(samples):
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    t0  = time.time()
    with torch.no_grad():
        out = model.generate(**inp,
                             max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False,
                             use_cache=False)      
    dt   = time.time() - t0
    gen  = out.shape[-1] - inp["input_ids"].shape[-1]
    lat  = dt / gen
    thr  = gen / dt
    lat_sum += lat
    thr_sum += thr
    if i < 10:
        wandb.log(dict(sample=i, latency_per_token=lat, throughput=thr))

enc = tokenizer("\n\n".join(samples), return_tensors="pt").to(model.device)
with torch.no_grad():
    loss = model(**enc, labels=enc["input_ids"]).loss
ppl  = math.exp(loss.item())

wandb.log(dict(avg_latency_per_token = lat_sum/len(samples),
               avg_throughput       = thr_sum/len(samples),
               perplexity           = ppl))
print(f"[Eval done] 8‑bit + LoRA, KV‑cache OFF  |  Perplexity {ppl:.2f}")
wandb.finish()
