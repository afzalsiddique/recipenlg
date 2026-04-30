"""
Conditional generation with the fine-tuned RecipeNLG GPT-2.

Two modes:
  1. --interactive: original REPL behavior (comma-separated ingredients,
     semicolon to close the list).
  2. --input_jsonl: read gold_test_100.json (output of prepare_text.py),
     generate --num_return_sequences recipes per gold prompt, write a flat
     JSON list of generated full-text recipes (in gold-major order) to
     --output_json. This is the format consumed by eval/replicate_metrics.py.

All decoding hyperparameters are listed explicitly via argparse.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging_setup import setup_logging  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", type=Path,
                   default=PROJECT_ROOT / "generation" / "model_out")
    p.add_argument("--input_jsonl", type=Path, default=None,
                   help="gold_test_100.json from prepare_text.py.")
    p.add_argument("--output_json", type=Path,
                   default=PROJECT_ROOT / "generation" / "data" / "generated_recipenlg.json")
    p.add_argument("--interactive", action="store_true")
    # Decoding HPs
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_return_sequences", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=950)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def build_prompt(ner_list) -> str:
    return "<RECIPE_START> <INPUT_START> " + " <NEXT_INPUT> ".join(ner_list) + " <INPUT_END>"


def decode_full_text(tokenizer, generated_ids, end_token_id) -> str:
    ids = generated_ids.tolist()
    if end_token_id in ids:
        ids = ids[: ids.index(end_token_id) + 1]
    text = tokenizer.decode(ids, clean_up_tokenization_spaces=True)
    return text


def render_markdown(full_text: str) -> str:
    md = re.sub(r"<RECIPE_(START|END)>", "", full_text)
    parts = md.split("<TITLE_START>")
    title = ""
    if len(parts) > 1:
        title = "# " + parts[1].replace("<TITLE_END>", "") + " #\n"
        md = parts[0]
    md = (md.replace("<INPUT_START>", "## Input ingredients ##\n`")
            .replace("<INPUT_END>", "`\n")
            .replace("<NEXT_INPUT>", "`\n`")
            .replace("<INGR_START>", "## Ingredients ##\n* ")
            .replace("<NEXT_INGR>", "\n* ")
            .replace("<INGR_END>", "\n")
            .replace("<INSTR_START>", "## Instructions ##\n1) ")
            .replace("<NEXT_INSTR>", "\n1) ")
            .replace("<INSTR_END>", "\n"))
    md = re.sub(r"( +`|` +)", "`", md)
    return title + md


def main() -> int:
    args = parse_args()
    log = setup_logging("run_generation")
    log.info("Args: %s", json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    log.info("Device=%s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    model_max_positions = int(getattr(model.config, "n_positions",
                                      getattr(model.config, "max_position_embeddings", 1024)))
    log.info("Model max positions: %d", model_max_positions)

    end_token_id = tokenizer.convert_tokens_to_ids("<RECIPE_END>")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if pad_id == end_token_id:
        log.warning("pad_id == <RECIPE_END> id (%d); generation will still work but "
                    "this matches the pre-fix-(B) training setup.", pad_id)

    gen_kwargs = dict(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        eos_token_id=end_token_id,
        pad_token_id=pad_id,
    )
    log.info("Decoding kwargs: %s", gen_kwargs)

    def call_kwargs(prompt_len: int) -> dict:
        budget = model_max_positions - prompt_len
        if budget <= 0:
            return None
        kw = dict(gen_kwargs)
        if budget < kw["max_new_tokens"]:
            kw["max_new_tokens"] = budget
        return kw

    if args.interactive:
        while True:
            raw = input("Comma-separated ingredients, semicolon to close >>> ")
            ner = [t.strip() for t in raw.replace(";", "").split(",") if t.strip()]
            prompt = build_prompt(ner)
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            kw = call_kwargs(ids.shape[1])
            if kw is None:
                print(f"Prompt has {ids.shape[1]} tokens, exceeds model max {model_max_positions}; skipping.")
                continue
            with torch.no_grad():
                out = model.generate(ids, **kw)
            for i in range(out.shape[0]):
                text = decode_full_text(tokenizer, out[i, ids.shape[1]:], end_token_id)
                print("=" * 60)
                print(render_markdown(prompt + text))
            if not raw:
                break
        return 0

    if args.input_jsonl is None:
        log.error("Either --interactive or --input_jsonl must be set.")
        return 2

    gold = json.loads(Path(args.input_jsonl).read_text())
    log.info("Loaded %d gold prompts from %s", len(gold), args.input_jsonl)

    generations = []
    for k, row in enumerate(gold):
        prompt = build_prompt(row["ner"])
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = ids.shape[1]
        kw = call_kwargs(prompt_len)
        if kw is None:
            log.warning("Skipping prompt %d: %d tokens >= model max %d",
                        k, prompt_len, model_max_positions)
            continue
        if kw["max_new_tokens"] != gen_kwargs["max_new_tokens"]:
            log.info("Prompt %d: clamping max_new_tokens %d -> %d (prompt_len=%d)",
                     k, gen_kwargs["max_new_tokens"], kw["max_new_tokens"], prompt_len)
        with torch.no_grad():
            out = model.generate(ids, **kw)
        for i in range(out.shape[0]):
            full_ids = out[i].tolist()
            if end_token_id in full_ids:
                full_ids = full_ids[: full_ids.index(end_token_id) + 1]
            text = tokenizer.decode(full_ids, clean_up_tokenization_spaces=True)
            generations.append(text)
        if (k + 1) % 10 == 0 or k + 1 == len(gold):
            log.info("Generated %d/%d gold prompts (%d total recipes)",
                     k + 1, len(gold), len(generations))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(generations, indent=2))
    log.info("Wrote %d generated recipes to %s", len(generations), args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
