"""
Build the GPT-2 fine-tuning corpus from RecipeNLG's full_dataset.csv.

Mirrors the original dataset2text.ipynb logic: drop rows with very short
titles or instructions, drop rows whose instructions contain "step" or
"mix all" (paper §5.2), do a 95/5 train/test split, and write the tagged
plain-text format consumed by tokenization.py.

Also writes gold_test_100.json: 100 random rows from the test split with
their NER-extracted entity list, used as evaluation prompts in §5.3.

All hyperparameters and paths are listed explicitly in argparse below.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging_setup import setup_logging  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = Path("/mmfs1/projects/changhui.yan/m.afzalsiddique/datasets/RecipeNLG/full_dataset.csv")
DEFAULT_OUT_DIR = PROJECT_ROOT / "generation" / "data"

STEP_REGEX = re.compile(r"(step|mix all)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train_filename", default="unsupervised_train.txt")
    p.add_argument("--test_filename", default="unsupervised_test.txt")
    p.add_argument("--gold_filename", default="gold_test_100.json")
    p.add_argument("--max_rows", type=int, default=-1,
                   help="If >0, only read the first N rows (for smoke tests).")
    p.add_argument("--test_size", type=float, default=0.05,
                   help="Fraction held out for the test split (paper: 5%).")
    p.add_argument("--n_gold", type=int, default=100,
                   help="Number of test recipes to sample as the gold standard.")
    p.add_argument("--min_title_len", type=int, default=4)
    p.add_argument("--min_directions_count", type=int, default=2)
    p.add_argument("--min_directions_chars", type=int, default=30)
    p.add_argument("--min_ingredients_chars", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_json_field(s):
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def filter_dataframe(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.copy()
    df["_title"] = df["title"].fillna("").astype(str)
    df["_directions"] = df["directions"].apply(parse_json_field)
    df["_ingredients"] = df["ingredients"].apply(parse_json_field)
    df["_ner"] = df["NER"].apply(parse_json_field)

    keep = (
        (df["_title"].str.len() >= args.min_title_len)
        & df["_directions"].apply(lambda x: isinstance(x, list) and len(x) >= args.min_directions_count
                                  and len("".join(x)) >= args.min_directions_chars)
        & df["_ingredients"].apply(lambda x: isinstance(x, list) and len("".join(x)) >= args.min_ingredients_chars)
        & df["_ner"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    )
    df_clean = df[keep].copy()

    has_step_or_mixall = df_clean["_directions"].apply(
        lambda dirs: bool(STEP_REGEX.search(" ".join(dirs)))
    )
    df_clean = df_clean[~has_step_or_mixall].copy()
    df_clean.reset_index(drop=True, inplace=True)
    return df_clean


def format_recipe(row) -> str:
    return (
        "<RECIPE_START> <INPUT_START> "
        + " <NEXT_INPUT> ".join(row["_ner"])
        + " <INPUT_END> <INGR_START> "
        + " <NEXT_INGR> ".join(row["_ingredients"])
        + " <INGR_END> <INSTR_START> "
        + " <NEXT_INSTR> ".join(row["_directions"])
        + " <INSTR_END> <TITLE_START> "
        + row["_title"]
        + " <TITLE_END> <RECIPE_END>"
    )


def write_text(df: pd.DataFrame, path: Path, log) -> None:
    log.info("Writing %d recipes to %s", len(df), path)
    with open(path, "w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 100000 == 0 and i:
                log.info("  ... wrote %d", i)
            f.write(format_recipe(row).replace("\n", " ") + "\n")


def main() -> int:
    args = parse_args()
    log = setup_logging("prepare_text")
    log.info("Args: %s", vars(args))
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reading %s", args.input_csv)
    read_kwargs = {"encoding": "utf-8"}
    if args.max_rows > 0:
        read_kwargs["nrows"] = args.max_rows
    df = pd.read_csv(args.input_csv, **read_kwargs)
    log.info("Loaded raw rows: %d, cols: %s", len(df), list(df.columns))

    df_clean = filter_dataframe(df, args)
    log.info("Kept %d / %d rows after filtering", len(df_clean), len(df))

    train_df, test_df = train_test_split(df_clean, test_size=args.test_size, random_state=args.seed)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    log.info("Split: train=%d test=%d", len(train_df), len(test_df))

    train_path = args.output_dir / args.train_filename
    test_path = args.output_dir / args.test_filename
    write_text(train_df, train_path, log)
    write_text(test_df, test_path, log)

    n_gold = min(args.n_gold, len(test_df))
    gold_df = test_df.sample(n=n_gold, random_state=args.seed).reset_index(drop=True)
    gold_records = []
    for _, row in gold_df.iterrows():
        gold_records.append({
            "title": row["_title"],
            "ingredients": row["_ingredients"],
            "directions": row["_directions"],
            "ner": row["_ner"],
            "full_text": format_recipe(row),
        })
    gold_path = args.output_dir / args.gold_filename
    gold_path.write_text(json.dumps(gold_records, indent=2))
    log.info("Wrote %d gold recipes to %s", len(gold_records), gold_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
