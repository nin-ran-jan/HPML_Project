import math, time, torch, wandb
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)
from peft import PeftModel                    

WANDB_PROJECT = "final_project"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
ADAPTER_DIR = "./llama2-7b-baseline-wikitext"      
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOAD_4BIT = True          
MAX_NEW_TOKENS = 50
MAX_TEST_SAMPLES = 100

wandb.init(
    project=WANDB_PROJECT,
    entity="ns3888-hpml",
    name="llama2-7b-ft-4bit-nokvcache",
    config=dict(quant="4bit" if LOAD_4BIT else "8bit",
                kv_cache=False, fine_tuned=True,
                max_new_tokens=MAX_NEW_TOKENS, samples=MAX_TEST_SAMPLES,
                device=DEVICE)
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=False)

bnb_cfg = BitsAndBytesConfig(
            load_in_4bit = LOAD_4BIT,
            load_in_8bit = not LOAD_4BIT,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
)

base = AutoModelForCausalLM.from_pretrained(
           BASE_MODEL,
           device_map="auto",
           quantization_config=bnb_cfg)

base.config.use_cache = False

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

ds      = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
samples = [s["text"] for s in ds.select(range(MAX_TEST_SAMPLES))
           if s["text"].strip()]

lat_sum = thr_sum = 0.0
for i, text in enumerate(samples):
    inp = tokenizer(text, return_tensors="pt").to(DEVICE)
    t0  = time.time()
    with torch.no_grad():
        outs = model.generate(**inp,
                              max_new_tokens=MAX_NEW_TOKENS,
                              do_sample=False,
                              use_cache=False)      
    dt   = time.time() - t0
    gen  = outs.shape[-1] - inp["input_ids"].shape[-1]
    lat  = dt / gen
    thr  = gen / dt
    lat_sum += lat;  thr_sum += thr

    if i < 10:
        wandb.log(dict(sample=i, latency_per_token=lat,
                       throughput=thr,
                       gen_preview=tokenizer.decode(outs[0],
                                                    skip_special_tokens=True)))

torch.cuda.empty_cache()         
model.config.use_cache = False    

enc = tokenizer("\n\n".join(samples), return_tensors="pt").to(DEVICE)
with torch.no_grad():
    loss = model(**enc, labels=enc["input_ids"]).loss
ppl  = math.exp(loss.item())

wandb.log(dict(avg_latency_per_token = lat_sum/len(samples),
               avg_throughput       = thr_sum/len(samples),
               test_perplexity      = ppl))
print(f"[Eval done] LoRA + {('4','8')[LOAD_4BIT is False]}‑bit, KV‑cache OFF "
      f"| Perplexity {ppl:.2f}")
wandb.finish()
