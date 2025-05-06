import os, time, math, warnings, torch
from types import MethodType

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.generation.candidate_generator import AssistedCandidateGenerator

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

warnings.filterwarnings("ignore", message="The attention mask.*", module="transformers")
warnings.filterwarnings("ignore", message="Setting pad_token_id.*",  module="transformers")

assert torch.cuda.is_available(), "needs a CUDA GPU"
DEVICE = "cuda"

class InstrumentedDraft(AssistedCandidateGenerator):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.generation_config.max_new_tokens = int(self.num_assistant_tokens)
        self.accepted = self.rejected = self.rollbacks = 0
        self.step_lat = []

    def get_candidates(self, input_ids):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        ids, logits = super().get_candidates(input_ids)
        end.record(); torch.cuda.synchronize()
        self.step_lat.append(start.elapsed_time(end))
        return ids, logits

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        super().update_candidate_strategy(input_ids, scores, num_matches)
        self.accepted += num_matches
        mism = scores.shape[1] - 1 - num_matches
        self.rejected += mism
        if mism:
            self.rollbacks += 1

MAIN_ID  = "meta-llama/Llama-3.1-8B"
DRAFT_ID = "meta-llama/Llama-3.2-1B"

print("Loading main model …")
main   = AutoModelForCausalLM.from_pretrained(
            MAIN_ID, torch_dtype=torch.float16, device_map="auto").eval()

print("Loading draft model …")
draft  = AutoModelForCausalLM.from_pretrained(
            DRAFT_ID, torch_dtype=torch.float16, device_map="auto").eval()

tok = AutoTokenizer.from_pretrained(MAIN_ID, use_fast=False)
tok.pad_token = tok.eos_token

GEN_TOKENS        = 64
NUM_ASSISTANT_TOK = 8

gen_cfg = GenerationConfig.from_model_config(main.config)
gen_cfg.max_new_tokens       = GEN_TOKENS
gen_cfg.num_assistant_tokens = NUM_ASSISTANT_TOK
gen_cfg.do_sample            = False

draft.generation_config.max_new_tokens      = NUM_ASSISTANT_TOK
draft.generation_config.num_assistant_tokens= NUM_ASSISTANT_TOK

prompt = "I am at a payphone trying to call home"
inputs = tok(prompt, return_tensors="pt", padding=True).to(DEVICE)

draft_gen = InstrumentedDraft(
    input_ids        = inputs.input_ids,
    assistant_model  = draft,
    generation_config= gen_cfg,
    model_kwargs     = {},
)

def _return_mine(self, **_): return draft_gen
main._get_candidate_generator = MethodType(_return_mine, main)

torch.cuda.synchronize(); t0 = time.time()
out = main.generate(**inputs, assistant_model=draft, generation_config=gen_cfg)
torch.cuda.synchronize(); total_t = time.time() - t0

cont_ids      = out[0][inputs.input_ids.shape[1]:]
generated_txt = tok.decode(cont_ids, skip_special_tokens=True)

with torch.no_grad():
    loss = main(cont_ids.unsqueeze(0), labels=cont_ids.unsqueeze(0)).loss
ppl = math.exp(loss.item())

n = cont_ids.numel()
stats = {
    "tokens_generated": n,
    "throughput_tok_s": n / total_t,
    "latency_sec_tok":  total_t / n,
    "accept_rate":      draft_gen.accepted / (draft_gen.accepted + draft_gen.rejected + 1e-9),
    "rollbacks":        draft_gen.rollbacks,
    "perplexity":       ppl,
}

print("\n=== Generation ===\n", generated_txt)
print("\n=== Metrics ===")
for k, v in stats.items():
    print(f"{k:18}: {v:.4f}" if isinstance(v, float) else f"{k:18}: {v}")