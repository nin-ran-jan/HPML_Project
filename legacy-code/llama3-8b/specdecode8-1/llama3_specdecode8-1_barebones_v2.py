from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device = {device}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to(device)

print("Models loaded")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, max_length=50, pad_token_id=tokenizer.eos_token_id )
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(outputs)
print(decoded)