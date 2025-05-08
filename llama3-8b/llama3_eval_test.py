import os, time, math, torch, wandb, warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm

warnings.filterwarnings("ignore", message="The attention mask.*", module="transformers")
warnings.filterwarnings("ignore", message="Setting pad_token_id.*",  module="transformers")

DEVICE = "cuda"
assert torch.cuda.is_available(), "CUDA GPU required"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DATASET = "wikitext"
CONF_NAME = "wikitext-2-raw-v1"
MAX_PROMPT = 128
GEN_TOKENS = 64
NUM_SAMPLES = 50

def main():
    wandb.init(
        project="final_project",
        entity="ns3888-hpml",
        name="llama3_baseline_eval",
        config=dict(
            model=MODEL_ID,
            dataset=DATASET,
            gen_tokens=GEN_TOKENS,
            samples=NUM_SAMPLES,
        ),
    )

    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto").eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = GEN_TOKENS
    gen_cfg.do_sample = True  # or False for greedy
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.use_cache = True

    print("Loading dataset …")
    ds = load_dataset(DATASET, CONF_NAME, split="test")
    ds = ds.filter(lambda ex: len(tokenizer(ex["text"]).input_ids) >= MAX_PROMPT)

    if NUM_SAMPLES > 0:
        texts = [ex["text"] for ex in ds.select(range(NUM_SAMPLES)) if ex["text"].strip()]
    else:
        texts = [ex["text"] for ex in ds if ex["text"].strip()]

    print(f"Loaded {len(texts)} samples")

    total_tok, total_time = 0, 0

    print("Running baseline decoding …")
    for i, txt in tqdm(enumerate(texts), total=len(texts)):
        prompt = " ".join(txt.strip().replace("\n", " ").split()[:MAX_PROMPT])
        if len(prompt) < 5:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, max_length=MAX_PROMPT, truncation=True).to(DEVICE)

        torch.cuda.synchronize(); t0 = time.time()
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=gen_cfg,
        )
        torch.cuda.synchronize(); dt = time.time() - t0

        cont_ids = out[0][inputs.input_ids.shape[1]:]
        n = cont_ids.numel()
        if n == 0: continue

        total_tok += n
        total_time += dt

        wandb.log({
            "sample_id": i,
            "tokens_generated": n,
            "latency_per_token": dt / n,
            "throughput": n / dt,
        })

    print("\n=== Final Averages ===")
    print(f"Latency/token (s): {total_time / total_tok:.4f}")
    print(f"Throughput (tok/s): {total_tok / total_time:.2f}")

    wandb.log({
        "avg_latency_per_token": total_time / total_tok,
        "avg_throughput": total_tok / total_time,
    })

    wandb.finish()

if __name__ == "__main__":
    main()
