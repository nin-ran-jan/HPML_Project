import time, argparse, wandb, torch, copy
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from transformers.generation.candidate_generator import CandidateGenerator

# use bash script to specify the parameters you want when you run this code
arg = argparse.ArgumentParser()

# specify target and draft model here
arg.add_argument("model",              type=str)
arg.add_argument("--aux-model",        type=str, required=True)
arg.add_argument("--dtype",            type=str, default="bf16")
# specify if you want quantization for the target and draft models and to what precision if so
arg.add_argument("--target-quant",        type=str, choices=["none","8bit","4bit"], default="none")
arg.add_argument("--draft-quant",         type=str, choices=["none","8bit","4bit"], default="none")
# specify the number of samples you want to run on
arg.add_argument("--num-samples",      type=int, default=500)
# specifies the max prompt length so it will clip any prompts beyond this value
arg.add_argument("--max-prompt",       type=int, default=128)
# specifies how many tokens will be generated 
arg.add_argument("--gen-toks",         type=int, default=128)
# specifies the number of tokens the draft model will generate and propose to the target model
arg.add_argument("--assist-toks",      type=int, default=8)
# the logit confidence threshold to early reject assistant tokens 
arg.add_argument("--assistant-confidence-threshold",      type=float, default=0.2)
arg.add_argument("--compile",          action="store_true",
                 help="torch.compile() the target")
# either going to be greedy or sampling
arg.add_argument("--do-sample",          action="store_true",
                 help="sample for the model instead of greedy")

arg.add_argument("--wandb-project",    type=str, default="final_project")
arg.add_argument("--wandb-entity",     type=str, default="ns3888-hpml")
arg.add_argument("--wandb-run",        type=str, default=None)
args = arg.parse_args()

dtype = {"fp16":torch.float16,"float16":torch.float16,
         "bf16":torch.bfloat16,"bfloat16":torch.bfloat16}.get(
            args.dtype.lower(), torch.float32)
device = "cuda";  assert torch.cuda.is_available(), "GPU required"

wandb.init(project=args.wandb_project,
           entity =args.wandb_entity,
           name   =args.wandb_run,
           config =vars(args))

# bitsandbytes quantization
def make_bnb(quant):
    if quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return None

print("Loading target …")
t_kwargs = {"device_map":"auto"}
bnb_t = make_bnb(args.target_quant)
if bnb_t:
    t_kwargs["quantization_config"] = bnb_t
    t_kwargs["torch_dtype"] = torch.float16  
else:
    t_kwargs["torch_dtype"] = dtype
target = AutoModelForCausalLM.from_pretrained(args.model, **t_kwargs).eval()

print("Loading draft  …")
d_kwargs = {"device_map":{"":0}}   
bnb_d = make_bnb(args.draft_quant)
if bnb_d:
    d_kwargs["quantization_config"] = bnb_d
    d_kwargs["torch_dtype"] = torch.float16
else:
    d_kwargs["torch_dtype"] = dtype
draft = AutoModelForCausalLM.from_pretrained(args.aux_model, **d_kwargs).eval()

if args.compile and args.target_quant == "none":
    target = torch.compile(target, mode="reduce-overhead")
if args.compile and args.draft_quant == "none":
    draft  = torch.compile(draft , mode="reduce-overhead")

tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
tok.pad_token = tok.eos_token

# subclass implemented to overload methods when necessary and keep track of key metrics such as rollbacks and acceptance rate
class MeteredDraft(AssistedCandidateGenerator):
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        self.generation_config.num_assistant_tokens = args.assist_toks
        self.generation_config.max_length = (
            args.max_prompt + args.gen_toks + args.assist_toks)
        self.generation_config.num_assistant_tokens_schedule = "constant"
        self.generation_config.do_sample = args.do_sample
        self.accepted = self.rejected = self.rollbacks = 0
        self.assistant_confidence_threshold = args.assistant_confidence_threshold

    def update_candidate_strategy(self, ids, scores, nmatch):
        super().update_candidate_strategy(ids, scores, nmatch)
        self.accepted += nmatch
        bad = scores.shape[1]-1-nmatch
        self.rejected += bad
        if bad: self.rollbacks += 1

    # Monkey patch, fix more elegentantly later
    def _calculate_new_tokens(self, input_ids: torch.LongTensor) -> Tuple[int, int]:
        """Calculate the minimum and maximum number of new tokens to generate."""
        self.num_assistant_tokens = args.assist_toks
        return super()._calculate_new_tokens(input_ids)

    # Monkey patch, fix more elegentantly later
    def _generate_candidates(self, generation_args: Dict) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """ 
        Patching to prevent threshold from early stopping draft generation to match orig paper 
        Also setting threshold each time to prevent auto scaling of threshold
        """
        generation_args["assistant_confidence_threshold"] = self.assistant_confidence_threshold
        return super()._generate_candidates(generation_args)

print("Loading WikiText-2 …")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
ds = ds.filter(lambda ex: len(tok(ex["text"]).input_ids) >= args.max_prompt)
if args.num_samples is not None and args.num_samples > 0:
    texts = ds.select(range(args.num_samples))["text"]
else:                              
    texts = ds["text"]  
print(f"Running on {len(texts)} samples")

prompts = []
for raw in texts:
    p = " ".join(raw.replace("\n"," ").split()[:args.max_prompt])
    t = tok(p, return_tensors="pt", padding=True,
            max_length=args.max_prompt, truncation=True)
    prompts.append({k:v.pin_memory() for k,v in t.items()})

base_cfg = copy.deepcopy(target.generation_config)
base_cfg.do_sample = args.do_sample
base_cfg.num_assistant_tokens = args.assist_toks
base_cfg.pad_token_id = tok.pad_token_id # possible fix for warning


#warm up
dummy = {k:v.to(device) for k,v in prompts[0].items()}
with torch.inference_mode():
    target.generate(**dummy, generation_config=base_cfg, max_new_tokens=2)
    draft.generate(**dummy, generation_config=base_cfg, max_new_tokens=2)

def run_loop(spec_decode: bool):
    tot_tok=tot_time=acc=rej=rb=0
    for i, prm in tqdm(enumerate(prompts), total=len(prompts),
                       desc="ASSISTED" if spec_decode else "BASELINE"):
        inp = {k:v.to(device, non_blocking=True) for k,v in prm.items()}
        gen_cfg = base_cfg     

        assist_args={}
        if spec_decode:
            draft_gen = MeteredDraft(inp["input_ids"], assistant_model=draft,
                                     generation_config=gen_cfg, model_kwargs={})
            target._get_candidate_generator = lambda *_ , **__: draft_gen
            assist_args["assistant_model"] = draft
            assist_args["num_assistant_tokens"] = args.assist_toks

        torch.cuda.synchronize(); t0=time.time()
        with torch.inference_mode():
            out = target.generate(**inp, generation_config=gen_cfg, **assist_args)
        torch.cuda.synchronize(); dt=time.time()-t0

        cont = out.shape[1]-inp["input_ids"].shape[1]
        if cont==0: continue
        tot_tok  += cont
        tot_time += dt
        if spec_decode:
            acc += draft_gen.accepted
            rej += draft_gen.rejected
            rb  += draft_gen.rollbacks
            wandb.log({"lat_tok_assisted": dt/cont,
                       "thr_assisted":     cont/dt,
                       "accept_rate_sample":
                           draft_gen.accepted/max(draft_gen.accepted+draft_gen.rejected,1)})

    m = {"tok":tot_tok,"time":tot_time,"lat":tot_time/tot_tok,"thr":tot_tok/tot_time}
    if spec_decode:
        m |= {"acc":acc,"rej":rej,"rb":rb,
              "accrate":acc/max(acc+rej,1)}
    return m

assist   = run_loop(True)
baseline = run_loop(False)

# baseline = { # from our baseline run on Llama3.1-8b
#     "lat": 0.06506,
#     "thr": 15.37051,
# }


spd = baseline["lat"]/assist["lat"]
print("\n=== RESULTS ===")
print(f"Baseline  : {baseline['lat']:.4f} s/tok | {baseline['thr']:.2f} tok/s")
print(f"Assisted  : {assist['lat']:.4f} s/tok | {assist['thr']:.2f} tok/s")
print(f"Speed-up  : {spd:.2f}×")
print(f"Accept-rate: {assist['accrate']:.2f} (roll-backs {assist['rb']})")

wandb.log({
    "baseline_latency_tok": baseline["lat"],
    "baseline_thr":         baseline["thr"],
    "assisted_latency_tok": assist["lat"],
    "assisted_thr":         assist["thr"],
    "speedup_ratio":        spd,
    "accept_rate":          assist["accrate"],
    "total_rollbacks":      assist["rb"],
})

wandb.finish()