import time, torch, wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
PROMPT = "What are the benefits of using transformer‑based language models?"
MAX_NEW_TOKENS = 100
USE_KV_CACHE = True           

wandb.init(
    project="final_project",
    name=f"llama2-7b-inference-{'kvcache' if USE_KV_CACHE else 'nocache'}",
    config={
        "model": BASE_MODEL,
        "prompt": PROMPT[:64] + "...",
        "max_new_tokens": MAX_NEW_TOKENS,
        "kv_cache": USE_KV_CACHE,
    },
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
model.config.use_cache = USE_KV_CACHE       

input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)

torch.cuda.synchronize(); t0 = time.time()
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    use_cache=USE_KV_CACHE,                 
)
torch.cuda.synchronize(); dt = time.time() - t0

tok_s   = MAX_NEW_TOKENS / dt
result  = tokenizer.decode(outputs[0], skip_special_tokens=True)

wandb.log({"latency_sec": dt, "tokens_per_sec": tok_s, "output": result})
print(result)
print(f"\nKV‑cache {USE_KV_CACHE} | {tok_s:.2f} tok/s ({dt:.2f}s total)")

wandb.finish()
