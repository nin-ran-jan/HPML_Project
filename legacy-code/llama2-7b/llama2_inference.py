import torch
import time
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
PROMPT = "What are the benefits of using transformer-based language models?"
MAX_NEW_TOKENS = 100

wandb.init(
    project="final_project",
    name="llama2-7b-inference",
    config={
        "model": BASE_MODEL,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": False,
        "quantization": "none"
    }
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)

torch.cuda.synchronize()
start_time = time.time()

output = model.generate(
    input_ids=input_ids,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    use_cache=False
)

torch.cuda.synchronize()
end_time = time.time()

latency = end_time - start_time
per_token_latency = latency / MAX_NEW_TOKENS
throughput = MAX_NEW_TOKENS / latency
generated = tokenizer.decode(output[0], skip_special_tokens=True)

wandb.log({
    "latency_sec": latency,
    "per_token_latency_sec": per_token_latency,
    "throughput_tokens_per_sec": throughput,
    "output_text": generated
})

print(f"\n=== Output ===\n{generated}")
print(f"\nLatency: {latency:.2f}s for {MAX_NEW_TOKENS} tokens "
      f"({per_token_latency:.3f}s/token, {throughput:.2f} tok/s)")

wandb.finish()
