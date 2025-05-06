from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B",
    assistant_model="meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16
)

pipe_output = pipe("Once upon a time, ", max_new_tokens=50, do_sample=False)

print(pipe_output[0]["generated_text"])