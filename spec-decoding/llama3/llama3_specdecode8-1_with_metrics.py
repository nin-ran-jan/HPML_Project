from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from llama3_draft_with_metrics import InstrumentedDraft

main = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
draft = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-1B")

gen_cfg = GenerationConfig.from_model_config(main.config)
gen_cfg.num_assistant_tokens = 8

draft_gen = InstrumentedDraft(
    input_ids      = None,            
    assistant_model= draft,
    generation_config=gen_cfg,
    model_kwargs   = {},              
)

prompt = "Research shows that speculative decoding"
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
inputs = tok(prompt, return_tensors="pt").to(main.device)

out = main.generate(**inputs,
                    assistant_model=draft,          
                    candidate_generator=draft_gen,  
                    max_new_tokens=64)

stats = {
    "tokens": len(out[0]) - inputs.input_ids.shape[1],
    "tok/sec": (len(out[0]) - inputs.input_ids.shape[1]) /
               (sum(draft_gen.latencies)/1e3),
    "accept_rate": draft_gen.accepted / (draft_gen.accepted + draft_gen.rejected + 1e-9),
    "rollbacks": draft_gen.rollbacks,
}
print(stats)
