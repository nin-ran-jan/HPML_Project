from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

DATASET = "wikitext"
CONF_NAME = "wikitext-2-v1"
SPLIT = "test"
MODEL_ID = "meta-llama/Llama-3.1-8b"

ds = load_dataset(DATASET, CONF_NAME, split=SPLIT)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

texts = [x["text"].strip() for x in ds]
non_empty_texts = [t for t in texts if t]

char_lengths = [len(t) for t in non_empty_texts]
token_lengths = [len(tok(t)["input_ids"]) for t in non_empty_texts]

print(f"Total samples        : {len(ds)}")
print(f"Non-empty samples    : {len(non_empty_texts)}")
print(f"Avg chars per sample : {np.mean(char_lengths):.2f}")
print(f"Avg tokens per sample: {np.mean(token_lengths):.2f}")
print(f"Min tokens per sample: {np.min(token_lengths):.2f}")
print(f"Max tokens per sample: {np.max(token_lengths):.2f}")
print(f"Median tokens/sample : {np.median(token_lengths)}")

# print the smallest and largest samples
print("Smallest sample: ", non_empty_texts[np.argmin(token_lengths)])
print("Largest sample: ", non_empty_texts[np.argmax(token_lengths)])
print("Smallest sample length: ", np.min(token_lengths))
print("Largest sample length: ", np.max(token_lengths))

# sort the input texts by their token lengths
sorted_texts = sorted(non_empty_texts, key=lambda x: len(tok(x)["input_ids"]))

# print how many sameples have length >= x
print("Samples with length >= 128: ", len([t for t in sorted_texts if len(tok(t)["input_ids"]) >= 128]))

plt.hist(token_lengths, bins=50, color='skyblue')
plt.title("Token Count Distribution per Sample")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("token_count_distribution.png")
