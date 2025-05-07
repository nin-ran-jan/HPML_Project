import os, time, math, torch, wandb, warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from types import MethodType
from tqdm import tqdm

def _patched_prepare_generation_args(self, input_ids, min_new, max_new):
    return {
        self.input_ids_key: input_ids,
        "generation_config": self.generation_config,
        "logits_processor": self.logits_processor,
        "min_new_tokens": min_new if min_new > 0 else None,
        "max_new_tokens": max_new if max_new > 0 else None,
    }

AssistedCandidateGenerator._prepare_generation_args = _patched_prepare_generation_args

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
NUM_SAMPLES = -1
# maybe vary this
NUM_ASSISTANT_TOK = TODO

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

def main():
    wandb.init(
        project="final_project",
        entity="ns3888-hpml",
        name="llama3_specdecode_eval_with_metrics_quantized_draft_4bit",
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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              
        bnb_4bit_compute_dtype=torch.float16,   
        bnb_4bit_use_double_quant=True          
    )

    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_ID,
        device_map="auto",
        quantization_config=bnb_config
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = GenerationConfig.from_model_config(main_model.config)
    gen_cfg.max_new_tokens = GEN_TOKENS
    gen_cfg.num_assistant_tokens = NUM_ASSISTANT_TOK
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
        out = main_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            assistant_model=draft_model, 
            generation_config=gen_cfg
        )
        torch.cuda.synchronize(); dt = time.time() - t0

        cont_ids = out[0][inputs.input_ids.shape[1]:]
        n = cont_ids.numel()
        if n == 0: continue

        # full_input = torch.cat([inputs.input_ids[0], cont_ids], dim=0).unsqueeze(0).to(DEVICE)

        # labels = full_input.clone()
        # labels[0, :inputs.input_ids.shape[1]] = -100

        # with torch.no_grad():
        #     loss = main_model(full_input, labels=labels).loss
        # ppl = math.exp(loss.item())


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
