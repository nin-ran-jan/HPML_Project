import os, time, math, torch, wandb, pynvml
from types import MethodType
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

TARGET_ID = "meta-llama/Llama-3.1-8B"
DRAFT_ID = "meta-llama/Llama-3.2-1B"
DATASET = "wikitext"
CONF_NAME = "wikitext-2-raw-v1"
MAX_PROMPT = 128
GEN_TOKENS = 64
NUM_SAMPLES = 100
BATCH_SIZE = 4
NUM_ASSISTANT_TOK = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gpu_stats():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    return mem.used / 1e9, util.gpu

def _patched_prepare_generation_args(self, input_ids, min_new, max_new):
    args = {
        self.input_ids_key: input_ids,
        "generation_config": self.generation_config,
        "logits_processor": self.logits_processor,
    }
    if min_new > 0:
        args["min_new_tokens"] = min_new
    if max_new > 0:
        args["max_new_tokens"] = max_new
    return args

AssistedCandidateGenerator._prepare_generation_args = _patched_prepare_generation_args
assert torch.cuda.is_available(), "CUDA GPU is required."

class InstrumentedDraft(AssistedCandidateGenerator):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.generation_config.max_new_tokens = int(self.num_assistant_tokens)
        self.accepted = self.rejected = self.rollbacks = 0

    def get_candidates(self, input_ids):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        ids, logits = super().get_candidates(input_ids)
        end.record(); torch.cuda.synchronize()
        return ids, logits

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        super().update_candidate_strategy(input_ids, scores, num_matches)
        self.accepted += num_matches
        mism = scores.shape[1] - 1 - num_matches
        self.rejected += mism
        if mism:
            self.rollbacks += 1

print("Loading main model ...")
main = AutoModelForCausalLM.from_pretrained(
    TARGET_ID, torch_dtype=torch.float16, device_map="auto").eval()
print("Loading draft model ...")
draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_ID, torch_dtype=torch.float16, device_map="auto").eval()
tok = AutoTokenizer.from_pretrained(TARGET_ID, use_fast=False)
tok.pad_token = tok.eos_token

gen_cfg = GenerationConfig.from_model_config(main.config)
gen_cfg.max_new_tokens = GEN_TOKENS
gen_cfg.num_assistant_tokens = NUM_ASSISTANT_TOK
gen_cfg.do_sample = False
draft.generation_config.max_new_tokens = NUM_ASSISTANT_TOK
draft.generation_config.num_assistant_tokens = NUM_ASSISTANT_TOK

print("Loading dataset…")
ds = load_dataset(DATASET, CONF_NAME, split="test")
texts = [ex["text"] for ex in ds.select(range(NUM_SAMPLES)) if ex["text"].strip()]

wandb.init(
    project="final_project",
    name="llama3_specdecode_manual",
    config=dict(
        target=TARGET_ID,
        draft=DRAFT_ID,
        dataset=DATASET,
        gen_tokens=GEN_TOKENS,
        samples=NUM_SAMPLES,
        prompt_len=MAX_PROMPT,
    ),
)

print("Beginning batched evaluation ...")
total_tokens = total_latency = total_thr = total_ppl = 0
total_accept = total_reject = total_rollbacks = 0

for b in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_prompts = []
    for txt in texts[b:b+BATCH_SIZE]:
        prompt = txt.strip().replace("\n", " ")
        if len(prompt.split()) < 5:
            continue
        prompt = " ".join(prompt.split()[:MAX_PROMPT])
        batch_prompts.append(prompt)

    if not batch_prompts:
        continue

    inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    draft_gen = InstrumentedDraft(
        input_ids=inputs.input_ids,
        assistant_model=draft,
        generation_config=gen_cfg,
        model_kwargs={},
    )

    def _return_mine(self, **_): return draft_gen
    main._get_candidate_generator = MethodType(_return_mine, main)

    torch.cuda.synchronize(); t0 = time.time()
    outputs = main.generate(**inputs, assistant_model=draft, generation_config=gen_cfg)
    torch.cuda.synchronize(); total_t = time.time() - t0

    for i in range(len(outputs)):
        cont_ids = outputs[i][inputs.input_ids.shape[1]:]
        n = cont_ids.numel()
        if n == 0:
            continue
        with torch.no_grad():
            loss = main(cont_ids.unsqueeze(0), labels=cont_ids.unsqueeze(0)).loss
        ppl = math.exp(loss.item())
        total_ppl += ppl
        total_tokens += n
        total_latency += total_t
        total_thr += n / total_t
        total_accept += draft_gen.accepted
        total_reject += draft_gen.rejected
        total_rollbacks += draft_gen.rollbacks

        wandb.log({
            "latency_per_token": total_t / n,
            "throughput": n / total_t,
            "perplexity": ppl,
            "accept_rate": draft_gen.accepted / (draft_gen.accepted + draft_gen.rejected + 1e-9),
            "rollbacks": draft_gen.rollbacks,
        })

avg_latency = total_latency / total_tokens
avg_throughput = total_thr / (len(texts) // BATCH_SIZE)
avg_ppl = total_ppl / (len(texts) // BATCH_SIZE)
accept_rate = total_accept / (total_accept + total_reject + 1e-9)

print("\n=== Metrics ===")
print(f"Avg Latency/token  : {avg_latency:.4f} s")
print(f"Avg Throughput     : {avg_throughput:.2f} tok/s")
print(f"Avg Perplexity     : {avg_ppl:.2f}")
print(f"Accept Rate        : {accept_rate:.2f}")
print(f"Total Rollbacks    : {total_rollbacks}")

wandb.log({
    "avg_latency_per_token": avg_latency,
    "avg_throughput": avg_throughput,
    "avg_perplexity": avg_ppl,
    "overall_accept_rate": accept_rate,
    "total_rollbacks": total_rollbacks
})
wandb.finish()
