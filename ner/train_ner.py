"""
Train a spaCy v3 NER model on the manually annotated ingredient lines from
ner/dataset/train_*.json. Replicates the NER setup of RecipeNLG (Bień et al.,
INLG 2020) §5.1.

All hyperparameters are listed explicitly at the top via argparse, even when
matching the spaCy default. Logs go to logs/{ts}_train_ner.log.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging_setup import setup_logging  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NER_DIR = PROJECT_ROOT / "ner"
DATASET_DIR = NER_DIR / "dataset"
DEFAULT_TRAIN_FILES = [DATASET_DIR / "train_0.json", DATASET_DIR / "train_1.json"]
DEFAULT_OUTPUT_DIR = NER_DIR / "model_v3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_files", nargs="+", type=Path, default=DEFAULT_TRAIN_FILES,
                   help="JSON files with [{content, entities:[[s,e,label],...]}, ...].")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to save the trained spaCy pipeline.")
    p.add_argument("--label", default="food", help="Single NER label to train.")
    p.add_argument("--n_iter", type=int, default=30, help="Number of training epochs.")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout for nlp.update.")
    p.add_argument("--batch_min", type=float, default=4.0,
                   help="Minimum minibatch size (compounding lower bound).")
    p.add_argument("--batch_max", type=float, default=32.0,
                   help="Maximum minibatch size (compounding upper bound).")
    p.add_argument("--batch_compound", type=float, default=1.001,
                   help="Compounding factor for minibatch growth.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--blank_lang", default="en", help="Base blank spaCy language.")
    return p.parse_args()


def load_examples(files):
    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            rows.extend(json.load(fh))
    return rows


def make_training_data(rows, label: str):
    out = []
    skipped = 0
    for r in rows:
        text = r["content"]
        spans = []
        for start, end, lab in r.get("entities", []):
            if lab != label:
                continue
            spans.append((start, end, lab))
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        non_overlapping = []
        last_end = -1
        for s, e, lab in spans:
            if s >= last_end:
                non_overlapping.append((s, e, lab))
                last_end = e
            else:
                skipped += 1
        out.append((text, {"entities": non_overlapping}))
    return out, skipped


def main() -> int:
    args = parse_args()
    log = setup_logging("train_ner")
    log.info("Args: %s", vars(args))

    random.seed(args.seed)
    spacy.util.fix_random_seed(args.seed)

    rows = load_examples(args.train_files)
    log.info("Loaded %d annotated lines from %d files.", len(rows), len(args.train_files))

    train_data, skipped = make_training_data(rows, args.label)
    log.info("Built %d training examples (skipped %d overlapping spans).", len(train_data), skipped)

    nlp = spacy.blank(args.blank_lang)
    ner = nlp.add_pipe("ner")
    ner.add_label(args.label)

    optimizer = nlp.initialize(get_examples=lambda: [
        Example.from_dict(nlp.make_doc(text), ann) for text, ann in train_data
    ])
    log.info("Initialized blank %s pipeline with NER label '%s'.", args.blank_lang, args.label)

    sizes = compounding(args.batch_min, args.batch_max, args.batch_compound)
    for epoch in range(1, args.n_iter + 1):
        random.shuffle(train_data)
        losses = {}
        for batch in minibatch(train_data, size=sizes):
            examples = [Example.from_dict(nlp.make_doc(t), ann) for t, ann in batch]
            nlp.update(examples, drop=args.dropout, losses=losses, sgd=optimizer)
        log.info("Epoch %d/%d: losses=%s", epoch, args.n_iter, losses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(args.output_dir)
    log.info("Saved trained pipeline to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
