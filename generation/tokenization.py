"""
Pack the tagged plain-text corpus produced by prepare_text.py into a single
HDF5 file with `train` and `test` matrices of shape (n_records, block_size)
for GPT-2 fine-tuning.

Multiple recipes are concatenated to fill each row up to block_size BPE
tokens; remaining slots are padded with the <RECIPE_END> token id (matches
the original tokenization.py from the RecipeNLG repo).
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from transformers import GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging_setup import setup_logging  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "generation" / "data"

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
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="Directory containing unsupervised_{train,test}.txt.")
    p.add_argument("--train_filename", default="unsupervised_train.txt")
    p.add_argument("--test_filename", default="unsupervised_test.txt")
    p.add_argument("--output_h5", default="unsupervised.h5")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--base_tokenizer", default="gpt2",
                   help="HF tokenizer id used as the BPE base.")
    p.add_argument("--lower_case", action="store_true",
                   help="Forwarded to GPT2Tokenizer(do_lower_case=...).")
    return p.parse_args()


def build_tokenizer(name: str, lower_case: bool):
    tokenizer = GPT2Tokenizer.from_pretrained(name, do_lower_case=lower_case)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<RECIPE_END>"
    return tokenizer


def pack_split(tokenizer, txt_path: Path, block_size: int, end_token_id: int, log) -> np.ndarray:
    rows = []
    last: list = []
    n_lines = 0
    with open(txt_path, "r", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            if n_lines % 10000 == 0:
                log.info("  read=%d packed=%d", n_lines, len(rows))
            tokens = tokenizer.tokenize(line.rstrip("\n"))
            if len(tokens) > block_size:
                continue
            ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(last) + len(ids) <= block_size:
                last.extend(ids)
            else:
                while len(last) < block_size:
                    last.append(end_token_id)
                rows.append(last)
                last = list(ids)
    if last:
        while len(last) < block_size:
            last.append(end_token_id)
        rows.append(last)
    log.info("%s: %d input lines → %d packed rows", txt_path.name, n_lines, len(rows))
    return np.asarray(rows, dtype=np.int64)


def main() -> int:
    args = parse_args()
    log = setup_logging("tokenization")
    log.info("Args: %s", vars(args))

    tokenizer = build_tokenizer(args.base_tokenizer, args.lower_case)
    end_token_id = tokenizer.convert_tokens_to_ids("<RECIPE_END>")
    log.info("Tokenizer vocab=%d  end_token_id=%d", len(tokenizer), end_token_id)

    out_path = args.data_dir / args.output_h5
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as hf:
        for split, fname in (("train", args.train_filename), ("test", args.test_filename)):
            txt_path = args.data_dir / fname
            if not txt_path.exists():
                log.warning("Skipping missing split %s (%s)", split, txt_path)
                continue
            mat = pack_split(tokenizer, txt_path, args.block_size, end_token_id, log)
            hf.create_dataset(split, data=mat)
            log.info("Wrote %s shape=%s to %s", split, mat.shape, out_path)
    log.info("Done. HDF5 file: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
