import os
import time
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
CHECKPOINT_DIR = "./mistral-baseline-wikitext-quant-final/checkpoint-100"
PROMPT = "What are the benefits of using transformer-based language models?"
MAX_NEW_TOKENS = 100

USE_KV_CACHE = True           

wandb.init(
    project="final_project",
    name=f"mistral‑7b‑inference‑{'kvcache' if USE_KV_CACHE else 'nocache'}",
    config={
        "base_model": BASE_MODEL,
        "adapter": CHECKPOINT_DIR,
        "prompt_preview": PROMPT[:64] + "...",
        "max_new_tokens": MAX_NEW_TOKENS,
        "kv_cache": USE_KV_CACHE,
    },
)

def load_model():
    """Load base Mistral‑7B and LoRA adapter, set KV‑cache flag."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    base_model.config.use_cache = USE_KV_CACHE

    peft_cfg = PeftConfig.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
    model = PeftModel.from_pretrained(
        base_model,
        model_id=CHECKPOINT_DIR,
        config=peft_cfg,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def run(prompt: str):
    tokenizer, model = load_model()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    torch.cuda.synchronize()
    t0 = time.time()

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=USE_KV_CACHE,        
    )

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    toks_per_sec = MAX_NEW_TOKENS / elapsed
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    wandb.log(
        {
            "latency_sec": elapsed,
            "tokens_per_sec": toks_per_sec,
            "output": generated_text,
        }
    )

    print("\n===== Generated =====\n")
    print(generated_text)
    print("\n---------------------")
    print(
        f"KV‑cache: {USE_KV_CACHE} | "
        f"Latency {elapsed:.2f}s for {MAX_NEW_TOKENS} tokens "
        f"→ {toks_per_sec:.2f} tok/s"
    )

if __name__ == "__main__":
    run(PROMPT)
    wandb.finish()
