import torch
import time
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
CHECKPOINT_DIR = "./mistral-baseline-wikitext-quant-final/checkpoint-100"
PROMPT = "What are the benefits of using transformer-based language models?"
MAX_NEW_TOKENS = 100

wandb.init(
    project="final_project",
    name="mistral-7b-eval-wikitext-subset-inference-quant",
    config={
        "model": BASE_MODEL,
        "adapter_path": CHECKPOINT_DIR,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": False,
        "quantization": "8-bit adapter (PEFT)",
        "speculative_decoding": False
    }
)

def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    config = PeftConfig.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
    model = PeftModel.from_pretrained(
        base_model,
        model_id=CHECKPOINT_DIR,
        config=config,
        is_trainable=False,
        local_files_only=True
    )
    model.eval()
    return tokenizer, model

def generate(tokenizer, model, prompt, max_new_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    torch.cuda.synchronize()
    start_time = time.time()

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False
    )

    torch.cuda.synchronize()
    end_time = time.time()

    latency = end_time - start_time
    per_token_latency = latency / max_new_tokens
    throughput = max_new_tokens / latency
    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    wandb.log({
        "latency_sec": latency,
        "per_token_latency_sec": per_token_latency,
        "throughput_tokens_per_sec": throughput,
        "output_text": generated
    })

    print(f"\n=== Output ===\n{generated}")
    print(f"\nLatency: {latency:.2f}s for {max_new_tokens} tokens "
          f"({per_token_latency:.3f}s/token, {throughput:.2f} tok/s)")

if __name__ == "__main__":
    tokenizer, model = load_finetuned_model()
    generate(tokenizer, model, PROMPT, MAX_NEW_TOKENS)
    wandb.finish()
