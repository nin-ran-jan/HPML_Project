import time
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MAX_NEW_TOKENS = 50
MAX_TEST_SAMPLES = 100
USE_KV_CACHE = True             

WANDB_PROJECT = "final_project"
MODEL_DIR = "./mistral-baseline-wikitext-quant-final"
CHECKPOINT_DIR = "./mistral-baseline-wikitext-quant-final/checkpoint-100"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

wandb.init(
    project=WANDB_PROJECT,
    name=f"mistral-7b-eval-wikitext-{'kvcache' if USE_KV_CACHE else 'nocache'}",
    config={
        "eval_split": f"test[:{MAX_TEST_SAMPLES}]",
        "eval_max_new_tokens": MAX_NEW_TOKENS,
        "device": DEVICE,
        "kv_cache": USE_KV_CACHE,        
    },
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
)

base_model.config.use_cache = USE_KV_CACHE    

model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.eval()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
dataset = dataset.select(range(MAX_TEST_SAMPLES))
samples = [s["text"] for s in dataset if s["text"].strip()]

total_latency, total_throughput = 0.0, 0.0

for i, prompt in enumerate(samples):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=USE_KV_CACHE, 
        )
    elapsed = time.time() - start

    gen_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    latency    = elapsed / gen_tokens
    throughput = gen_tokens / elapsed

    total_latency   += latency
    total_throughput += throughput

    if i < 10:
        wandb.log({
            "sample_id": i,
            "prompt": prompt[:200] + "...",
            "generation": tokenizer.decode(outputs[0], skip_special_tokens=True),
            "latency_per_token": latency,
            "throughput": throughput,
        })

# perplexity
encodings = tokenizer(" ".join(samples), return_tensors="pt").to(DEVICE)
labels = encodings["input_ids"]
with torch.no_grad():
    loss = model(**encodings, labels=labels).loss
    perplexity = torch.exp(loss).item()

wandb.log({
    "avg_latency_per_token": total_latency / len(samples),
    "avg_throughput": total_throughput / len(samples),
    "test_perplexity": perplexity,
})
print(
    f"[Eval Done] KV‑cache={USE_KV_CACHE} | "
    f"Perplexity: {perplexity:.2f} on {len(samples)} samples"
)

wandb.finish()
