import torch
import time
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

WANDB_PROJECT = "final_project"
MODEL_DIR = "llama2-7b-baseline-wikitext"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 50
MAX_TEST_SAMPLES = 100

wandb.init(
    project=WANDB_PROJECT,
    name="llama2-7b-eval-wikitext-subset",
    config={
        "eval_split": f"test[:{MAX_TEST_SAMPLES}]",
        "eval_max_new_tokens": MAX_NEW_TOKENS,
        "device": DEVICE
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", torch_dtype=torch.float16)
model.eval()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
dataset = dataset.select(range(MAX_TEST_SAMPLES))
samples = [sample["text"] for sample in dataset if sample["text"].strip()]

total_latency, total_throughput = 0, 0
for i, prompt in enumerate(samples):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False
        )
    end = time.time()

    gen_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    elapsed = end - start
    latency = elapsed / gen_tokens
    throughput = gen_tokens / elapsed

    total_latency += latency
    total_throughput += throughput

    if i < 10:
        wandb.log({
            "sample_id": i,
            "prompt": prompt[:200] + "...",
            "generation": tokenizer.decode(outputs[0], skip_special_tokens=True),
            "latency_per_token": latency,
            "throughput": throughput,
        })

encodings = tokenizer(" ".join(samples), return_tensors="pt").to(DEVICE)
labels = encodings["input_ids"]
with torch.no_grad():
    loss = model(**encodings, labels=labels).loss
    perplexity = torch.exp(loss).item()

wandb.log({
    "avg_latency_per_token": total_latency / len(samples),
    "avg_throughput": total_throughput / len(samples),
    "test_perplexity": perplexity
})
print(f"[Eval Done] Perplexity: {perplexity:.2f} | Samples: {len(samples)}")

wandb.finish()
