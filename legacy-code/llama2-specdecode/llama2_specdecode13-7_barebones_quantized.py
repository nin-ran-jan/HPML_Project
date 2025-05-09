from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

bnb_config = BitsAndBytesConfig(load_in_8bit=True, device_map="cuda")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

assistant_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

print("Models loaded")

inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

outputs = model.generate(**inputs, assistant_model=assistant_model)

decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Generated Text:\n")
for i, output in enumerate(decoded_outputs):
    print(f"[Sample {i+1}]: {output}\n")
