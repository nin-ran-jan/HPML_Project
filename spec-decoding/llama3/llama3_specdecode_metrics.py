import time, math, torch
from types import MethodType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.generation.candidate_generator import AssistedCandidateGenerator

DEVICE = "cuda"
class InstrumentedDraft(AssistedCandidateGenerator):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.accepted = 0
        self.rejected = 0
        self.rollbacks = 0
        self.step_lat = []         

    def get_candidates(self, input_ids):
        start = torch.cuda.Event(True)
        end   = torch.cuda.Event(True)
        start.record()

        ids, logits = super().get_candidates(input_ids)

        end.record(); torch.cuda.synchronize()
        self.step_lat.append(start.elapsed_time(end))  
        return ids, logits

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        super().update_candidate_strategy(input_ids, scores, num_matches)
        self.accepted += num_matches
        mismatch = scores.shape[1] - 1 - num_matches
        self.rejected += mismatch
        if mismatch:
            self.rollbacks += 1


MAIN_ID  = "meta-llama/Meta-Llama-3-8B"
DRAFT_ID = "meta-llama/Meta-Llama-3-1B"

print("Loading main model …")
main = AutoModelForCausalLM.from_pretrained(
    MAIN_ID, torch_dtype=torch.float16, device_map="auto"
).eval()

print("Loading draft model …")
draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_ID, torch_dtype=torch.float16, device_map="auto"
).eval()

tok = AutoTokenizer.from_pretrained(MAIN_ID, use_fast=False)

gen_cfg = GenerationConfig.from_model_config(main.config)
gen_cfg.num_assistant_tokens = 8      
gen_cfg.max_new_tokens       = 64
gen_cfg.do_sample            = False  
gen_cfg.is_assistant         = False  

prompt = "Research shows that speculative decoding"
inputs = tok(prompt, return_tensors="pt").to(DEVICE)

draft_gen = InstrumentedDraft(
    input_ids=inputs.input_ids,           
    assistant_model=draft,
    generation_config=gen_cfg,
    model_kwargs={},                    
)

def _return_my_generator(self, **_):
    return draft_gen

main._get_candidate_generator = MethodType(_return_my_generator, main)

torch.cuda.synchronize(); t0 = time.time()
out = main.generate(
    **inputs,
    assistant_model=draft,
    generation_config=gen_cfg,   
)
torch.cuda.synchronize(); total_time = time.time() - t0

gen_ids   = out[0][inputs.input_ids.shape[1]:]       
generated = tok.decode(gen_ids, skip_special_tokens=True)

with torch.no_grad():
    loss = main(gen_ids.unsqueeze(0), labels=gen_ids.unsqueeze(0)).loss
ppl = math.exp(loss.item())

tok_gen   = gen_ids.numel()
throughput = tok_gen / total_time
lat_tok    = total_time / tok_gen
acc_rate   = draft_gen.accepted / (draft_gen.accepted + draft_gen.rejected + 1e-9)

stats = {
    "tokens_generated": tok_gen,
    "throughput_tok_s": throughput,
    "latency_sec_tok":  lat_tok,
    "accept_rate":      acc_rate,
    "rollbacks":        draft_gen.rollbacks,
    "perplexity":       ppl,
}

print("\n=== Generation ===\n", generated)
print("\n=== Metrics ===")
for k, v in stats.items():
    print(f"{k:18}: {v:.4f}" if isinstance(v, float) else f"{k:18}: {v}")
