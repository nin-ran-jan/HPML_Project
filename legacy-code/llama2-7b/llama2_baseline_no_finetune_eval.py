import math, time, torch, wandb
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME        = "meta-llama/Llama-2-7b-hf"
WANDB_PROJECT     = "final_project"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS    = 50
MAX_TEST_SAMPLES  = 100
STRIDE            = 512                    

wandb.init(
    project=WANDB_PROJECT,
    name="llama2-7b-baseline-fair-eval",
    config=dict(device=DEVICE, stride=STRIDE,
                max_new_tokens=MAX_NEW_TOKENS,
                test_samples=MAX_TEST_SAMPLES),
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model     = AutoModelForCausalLM.from_pretrained(
               MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
model.eval()

ds       = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
samples  = [s["text"] for s in ds.select(range(MAX_TEST_SAMPLES)) if s["text"].strip()]

lat_sum = thr_sum = 0.0
for text in samples:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, use_cache=False)
    dt   = time.time() - t0
    gen  = out.shape[-1] - inputs["input_ids"].shape[-1]
    lat_sum += dt / gen
    thr_sum += gen / dt

wandb.log(dict(avg_latency_per_token = lat_sum / len(samples),
               avg_throughput       = thr_sum / len(samples)))

def fair_perplexity(model, tokenizer, text: str,
                    stride: int = 512, device="cuda"):

    if tokenizer.bos_token and not text.startswith(tokenizer.bos_token):
        text = tokenizer.bos_token + text

    enc   = tokenizer(text, return_tensors="pt")
    ids   = enc["input_ids"].to(device)
    seq_l = ids.size(1)
    ctx   = model.config.max_position_embeddings     

    nll, counted = 0.0, 0
    for start in tqdm(range(0, seq_l, stride), leave=False):
        end      = min(start + ctx, seq_l)
        trg_len  = end - start
        slice_ids = ids[:, start:end]
        targets   = slice_ids.clone()
        targets[:, :-trg_len] = -100               

        with torch.no_grad():
            logits = model(slice_ids).logits
            loss   = F.cross_entropy(
                       logits.view(-1, logits.size(-1)),
                       targets.view(-1),
                       ignore_index=-100,
                       reduction="sum")             
        nll      += loss.item()
        counted  += trg_len

    return math.exp(nll / counted)

full_text = "\n\n".join(samples)
ppl = fair_perplexity(model, tokenizer, full_text,
                      stride=STRIDE, device=model.device)

wandb.log(dict(perplexity = ppl))
print(f"Llama‑2 7B baseline perplexity (fair): {ppl:.2f}")
wandb.finish()
