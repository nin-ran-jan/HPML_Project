import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
import wandb

# Avoid tokenizer warnings when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load a model with the desired quantization level
def load_model(model_id, quant):
    if quant == "16":
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
    elif quant == "8":
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_8bit=True
        )
    elif quant == "4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config
        )
    else:
        raise ValueError("quant must be one of: '16', '8', or '4'")

# Generate a single code completion from the model
def generate_one_completion(model, tokenizer, prompt, max_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        # temperature=0.7,
        # top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):]  # Strip the prompt from the generated output

def main(args):
    # If no run name was provided, create one based on model + config
    if args.wandb_run is None:
        model_name = args.model_id.split("/")[-1].replace(".", "-")
        args.wandb_run = f"{model_name}-{args.quant}bit-{args.num_tasks}tasks-{args.num_samples}samples"

    # Initialize Weights & Biases logging
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        config=vars(args)
    )

    print(f"Loading model: {args.model_id} at {args.quant}-bit precision")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = load_model(args.model_id, args.quant)
    model.eval()

    # Load a subset of HumanEval problems (default: first 40)
    print(f"Loading first {args.num_tasks} HumanEval tasks...")
    problems = read_problems()
    limited_problems = list(problems.items())[:args.num_tasks]

    # Save the subset to a file for evaluation use
    print("Saving subset_problem_file.jsonl...")
    write_jsonl("subset_problem_file.jsonl", [
        {
            "task_id": k,
            "prompt": v["prompt"],
            "test": v["test"],
            "entry_point": v["entry_point"]
        }
        for k, v in limited_problems
    ])

    # Generate completions for each task
    print("Generating completions...")
    samples = []
    for task_id, problem in tqdm(limited_problems):
        for _ in range(args.num_samples):
            completion = generate_one_completion(model, tokenizer, problem["prompt"], args.max_tokens)
            samples.append({"task_id": task_id, "completion": completion})

    # Save the generated completions
    samples_path = f"{args.output_prefix}.jsonl"
    print(f"Writing completions to {samples_path}")
    write_jsonl(samples_path, samples)

    # Evaluate functional correctness via HumanEval harness
    print("Running evaluation...")
    results = evaluate_functional_correctness(
        sample_file=samples_path,
        k=[1, args.num_samples] if args.num_samples > 1 else [1],
        n_workers=None,
        timeout=3.0,
        problem_file="subset_problem_file.jsonl"
    )

    # Log pass@1 to wandb as a single summary metric
    wandb.log({"humaneval/pass@1": float(results["pass@1"])})

    wandb.finish()
    print(f"Results: {results}")

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and evaluate LLM completions on HumanEval")

    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID to load")
    parser.add_argument("--quant", type=str, choices=["16", "8", "4"], default="16", help="Quantization level")
    parser.add_argument("--num_tasks", type=int, default=40, help="Number of HumanEval tasks to run")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of completions per task")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--output_prefix", type=str, default="samples", help="Prefix for output JSONL files")
    parser.add_argument("--wandb_project", type=str, default="final_project", help="WandB project name")
    parser.add_argument("--wandb_entity",  type=str, default="ns3888-hpml", help="WandB entity/team")
    parser.add_argument("--wandb_run",     type=str, default=None, help="Optional name for this run")

    args = parser.parse_args()
    main(args)
