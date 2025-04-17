import torch
import time
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "What are the benefits of using transformer-based language models?"
MAX_NEW_TOKENS = 100

wandb.init(
    project="final_project",
    config={
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": False,
        "quantization": "None",
        "speculative_decoding": False
    }
)

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
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
    tokenizer, model = load_model()
    generate(tokenizer, model, PROMPT, MAX_NEW_TOKENS)
    wandb.finish()
