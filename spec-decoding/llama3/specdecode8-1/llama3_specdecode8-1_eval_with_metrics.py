import os, time, math, torch, wandb, warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from types import MethodType
from tqdm import tqdm
import copy


warnings.filterwarnings("ignore", message="The attention mask.*", module="transformers")
warnings.filterwarnings("ignore", message="Setting pad_token_id.*",  module="transformers")

DEVICE = "cuda"
assert torch.cuda.is_available(), "CUDA GPU required"

TARGET_ID = "meta-llama/Llama-3.1-8B"
DRAFT_ID = "meta-llama/Llama-3.2-1B"
DATASET = "wikitext"
CONF_NAME = "wikitext-2-raw-v1"
# maybe vary this
MAX_PROMPT = 128
GEN_TOKENS = 64
NUM_SAMPLES = 100
# maybe vary this
NUM_ASSISTANT_TOK = 8

class InstrumentedDraft(AssistedCandidateGenerator):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.generation_config.num_assistant_tokens = int(self.num_assistant_tokens)
        self.generation_config.max_length = MAX_PROMPT + GEN_TOKENS + NUM_ASSISTANT_TOK + 1
        self.accepted = self.rejected = self.rollbacks = 0
        self.generation_config.do_sample = True
        self.generation_config.num_assistant_tokens_schedule = "constant"
        # For decoder only models the max length includes prompt + generated tokens
        # self.generation_config.max_length =  MAX_PROMPT + GEN_TOKENS + NUM_ASSISTANT_TOK + 1
        # self.generation_config.do_sample = True

    def get_candidates(self, input_ids):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        ids, logits = super().get_candidates(input_ids)
        end.record(); torch.cuda.synchronize()
        generated_count = ids.shape[1] - input_ids.shape[1]
        # print(f">>> Draft generated {generated_count} tokens (expected: {self.num_assistant_tokens})")
        return ids, logits

    def update_candidate_strategy(self, input_ids, scores, num_matches):
        super().update_candidate_strategy(input_ids, scores, num_matches)
        self.accepted += num_matches
        mism = scores.shape[1] - 1 - num_matches
        self.rejected += mism
        if mism:
            self.rollbacks += 1

def main():
    wandb.init(
        project="final_project",
        entity="ns3888-hpml",
        name="llama3_specdecode8-1_eval_full_data_tok40",
        config=dict(
            target_model=TARGET_ID,
            draft_model=DRAFT_ID,
            dataset=DATASET,
            gen_tokens=GEN_TOKENS,
            samples=NUM_SAMPLES,
            assistant_tokens=NUM_ASSISTANT_TOK,
        ),
    )

    print("Loading main model …")
    main_model = AutoModelForCausalLM.from_pretrained(
        TARGET_ID, torch_dtype=torch.float16, device_map="auto").eval()

    print("Loading draft model …")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID, torch_dtype=torch.float16, device_map="auto").eval()
    draft_model.generation_config.num_assistant_tokens = NUM_ASSISTANT_TOK

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = GenerationConfig.from_model_config(main_model.config)
    # gen_cfg.max_new_tokens = GEN_TOKENS
    # gen_cfg.num_assistant_tokens = NUM_ASSISTANT_TOK
    gen_cfg.do_sample = True
    gen_cfg.pad_token_id = tokenizer.pad_token_id

    print("Loading dataset …")
    ds = load_dataset(DATASET, CONF_NAME, split="test")
    # sample a subset of the dataset with tokenized length >= MAX_PROMPT
    ds = ds.filter(lambda ex: len(tokenizer(ex["text"]).input_ids) >= MAX_PROMPT)

    if NUM_SAMPLES > 0:
        texts = [ex["text"] for ex in ds.select(range(NUM_SAMPLES)) if ex["text"].strip()]
    else:
        texts = [ex["text"] for ex in ds if ex["text"].strip()]

    print(f"Loaded {len(texts)} samples")

    total_tok, total_time = 0, 0
    total_accept, total_reject, total_rollback = 0, 0, 0

    print("Running evaluation …")
    for i, txt in tqdm(enumerate(texts), total=len(texts)):
        prompt = " ".join(txt.strip().replace("\n", " ").split()[:MAX_PROMPT])
        if len(prompt) < 5:
            continue

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)

        draft_gen = InstrumentedDraft(
            input_ids=inputs.input_ids,
            assistant_model=draft_model,
            generation_config=gen_cfg,
            model_kwargs={},
        )

        def _return_mine(self, **_): return draft_gen
        main_model._get_candidate_generator = MethodType(_return_mine, main_model)

        torch.cuda.synchronize(); t0 = time.time()
        # print("GENCFG",gen_cfg)
        out = main_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            assistant_model=draft_model, 
            generation_config=gen_cfg, 
            # use_model_defaults=False,
        )
        torch.cuda.synchronize(); dt = time.time() - t0

        cont_ids = out[0][inputs.input_ids.shape[1]:]
        n = cont_ids.numel()
        if n == 0: continue

        total_tok += n
        total_time += dt
        # total_ppl += ppl
        total_accept += draft_gen.accepted
        total_reject += draft_gen.rejected
        total_rollback += draft_gen.rollbacks

        wandb.log({
            "sample_id": i,
            "tokens_generated": n,
            "latency_per_token": dt / n,
            "throughput": n / dt,
            # "perplexity": ppl,
            "accept_rate": draft_gen.accepted / (draft_gen.accepted + draft_gen.rejected + 1e-9),
            "rollbacks": draft_gen.rollbacks,
        })

    print("\n=== Final Averages ===")
    print(f"Latency/token (s): {total_time / total_tok:.4f}")
    print(f"Throughput (tok/s): {total_tok / total_time:.2f}")
    # print(f"Perplexity: {total_ppl / NUM_SAMPLES:.2f}")
    print(f"Accept Rate: {total_accept / (total_accept + total_reject + 1e-9):.2f}")
    print(f"Total Rollbacks: {total_rollback}")

    wandb.log({
        "avg_latency_per_token": total_time / total_tok,
        "avg_throughput": total_tok / total_time,
        # "avg_perplexity": total_ppl / NUM_SAMPLES,
        "overall_accept_rate": total_accept / (total_accept + total_reject + 1e-9),
        "total_rollbacks": total_rollback
    })

    wandb.finish()

if __name__ == "__main__":
    main()