from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device = {device}")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)

print("Models loaded")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

outputs = model.generate(**inputs, assistant_model=assistant_model)
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(decoded)
