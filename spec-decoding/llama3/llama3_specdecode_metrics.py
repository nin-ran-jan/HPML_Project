import os, math, time, warnings, torch
from types import MethodType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.generation.candidate_generator import AssistedCandidateGenerator

warnings.filterwarnings(
    "ignore",
    message="The attention mask.*",
    module="transformers",
)
warnings.filterwarnings(
    "ignore",
    message="Setting pad_token_id to eos_token_id.*",
    module="transformers",
)

assert torch.cuda.is_available(), "Run on a CUDA-equipped machine"
DEVICE = "cuda"


class InstrumentedDraft(AssistedCandidateGenerator):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.accepted = self.rejected = self.rollbacks = 0
        self.step_lat = []        

    def get_candidates(self, input_ids):
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
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
main = AutoModelForCausalLM.from_pretrained(
    MAIN_ID, torch_dtype=torch.float16, device_map="auto"
).eval()

print("Loading draft model …")
draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_ID, torch_dtype=torch.float16, device_map="auto"
).eval()

tok = AutoTokenizer.from_pretrained(MAIN_ID, use_fast=False)
tok.pad_token = tok.eos_token
GEN_TOKENS          = 64
NUM_ASSISTANT_TOK   = 8

gen_cfg = GenerationConfig.from_model_config(main.config)
gen_cfg.max_new_tokens       = GEN_TOKENS
gen_cfg.num_assistant_tokens = NUM_ASSISTANT_TOK
gen_cfg.do_sample            = False      

draft.generation_config.num_assistant_tokens = NUM_ASSISTANT_TOK

prompt = "Research shows that speculative decoding"
inputs = tok(prompt, return_tensors="pt", padding=True).to(DEVICE)

draft_gen = InstrumentedDraft(
    input_ids        = inputs.input_ids,
    assistant_model  = draft,
    generation_config= gen_cfg,
    model_kwargs     = {},    
)

def _return_my_gen(self, **_):
    return draft_gen
main._get_candidate_generator = MethodType(_return_my_gen, main)

torch.cuda.synchronize(); t0 = time.time()
out = main.generate(
    **inputs,
    assistant_model   = draft,
    generation_config = gen_cfg,
)
torch.cuda.synchronize(); total_t = time.time() - t0

continuation_ids = out[0][inputs.input_ids.shape[1]:]
generated_text   = tok.decode(continuation_ids, skip_special_tokens=True)

with torch.no_grad():
    loss = main(continuation_ids.unsqueeze(0), labels=continuation_ids.unsqueeze(0)).loss
ppl = math.exp(loss.item())

tokens_gen = continuation_ids.numel()
stats = {
    "tokens_generated" : tokens_gen,
    "throughput_tok_s" : tokens_gen / total_t,
    "latency_sec_tok"  : total_t / tokens_gen,
    "accept_rate"      : draft_gen.accepted / (draft_gen.accepted + draft_gen.rejected + 1e-9),
    "rollbacks"        : draft_gen.rollbacks,
    "perplexity"       : ppl,
}

print("\n=== Generation ===\n", generated_text)
print("\n=== Metrics ===")
for k, v in stats.items():
    print(f"{k:18}: {v:.4f}" if isinstance(v, float) else f"{k:18}: {v}")
