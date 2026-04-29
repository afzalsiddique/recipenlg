"""
Fine-tune GPT-2 on the packed RecipeNLG corpus produced by tokenization.py.

This is a modernized rewrite of the original 2020 script: uses
AutoModelForCausalLM, the current HF Trainer API, and bf16/fp16 auto-detect
on the active GPU. All hyperparameters are listed explicitly via argparse.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging_setup import setup_logging  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPECIAL_TOKENS = [
    "<TITLE_START>",
    "<TITLE_END>",
    "<INSTR_START>",
    "<NEXT_INSTR>",
    "<INSTR_END>",
    "<INGR_START>",
    "<NEXT_INGR>",
    "<INGR_END>",
    "<RECIPE_START>",
    "<RECIPE_END>",
    "<INPUT_START>",
    "<INPUT_END>",
    "<NEXT_INPUT>",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Paths & data
    p.add_argument("--h5_path", type=Path,
                   default=PROJECT_ROOT / "generation" / "data" / "unsupervised.h5")
    p.add_argument("--output_dir", type=Path,
                   default=PROJECT_ROOT / "generation" / "model_out")
    p.add_argument("--model_name_or_path", default="gpt2",
                   help="HF model id or local path. Paper uses gpt2 (124M).")
    p.add_argument("--tokenizer_name_or_path", default=None,
                   help="If None, defaults to model_name_or_path.")
    # Block & data fraction
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--eval_split_offset", type=int, default=8100,
                   help="In the H5 'test' table, take rows [offset:] as the eval set "
                        "(matches the 30%% dev-set convention from the original repo).")
    # Optimization
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr_scheduler_type", default="linear")
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    # Logging / saving
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--eval_strategy", default="no",
                   choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=5000)
    p.add_argument("--report_to", default="none")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    # Misc
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite_output_dir", action="store_true")
    return p.parse_args()


class H5Dataset(Dataset):
    def __init__(self, h5_path: Path, split: str, eval_split_offset: int):
        with h5py.File(h5_path, "r") as fh:
            arr = fh[split][:]
        if split == "test":
            arr = arr[eval_split_offset:] if eval_split_offset < len(arr) else arr
        self.examples = arr

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def select_precision(log) -> dict:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            log.info("CUDA bf16 supported -> using bf16.")
            return {"bf16": True, "fp16": False}
        log.info("CUDA available but bf16 not supported -> using fp16.")
        return {"bf16": False, "fp16": True}
    log.info("CUDA not available -> running in fp32.")
    return {"bf16": False, "fp16": False}


def main() -> int:
    args = parse_args()
    log = setup_logging("run_lm_finetuning_new")
    log.info("Args: %s", json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<RECIPE_END>"
    log.info("Tokenizer vocab=%d pad_token=%r", len(tokenizer), tokenizer.pad_token)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    log.info("Loaded model %s, resized to %d embeddings", args.model_name_or_path, len(tokenizer))

    train_dataset = H5Dataset(args.h5_path, "train", args.eval_split_offset) if args.do_train else None
    eval_dataset = H5Dataset(args.h5_path, "test", args.eval_split_offset) if args.do_eval else None
    if train_dataset is not None:
        log.info("Train dataset rows=%d", len(train_dataset))
    if eval_dataset is not None:
        log.info("Eval dataset rows=%d", len(eval_dataset))

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    precision = select_precision(log)

    ta_kwargs = dict(
        output_dir=str(args.output_dir),
        do_train=args.do_train,
        do_eval=args.do_eval,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        **precision,
    )

    import inspect
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "overwrite_output_dir" in ta_params:
        ta_kwargs["overwrite_output_dir"] = args.overwrite_output_dir
    elif args.overwrite_output_dir and args.output_dir.exists():
        log.info("overwrite_output_dir not supported by this transformers version; "
                 "leaving %s in place (Trainer will overwrite checkpoints).", args.output_dir)
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = args.eval_strategy
    elif "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = args.eval_strategy

    training_args = TrainingArguments(**ta_kwargs)

    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    if args.do_train:
        log.info("Starting training")
        train_result = trainer.train()
        log.info("Training finished: %s", train_result.metrics)
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(args.output_dir)

    if args.do_eval and trainer.is_world_process_zero():
        log.info("Starting evaluation")
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            metrics["perplexity"] = math.exp(eval_loss)
        out = args.output_dir / "eval_results_lm.json"
        out.write_text(json.dumps(metrics, indent=2))
        log.info("Eval metrics: %s", metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
