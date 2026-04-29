"""
Evaluate the trained spaCy NER on ner/dataset/test_*.json using the RecipeNLG
penalty metric (paper §5.1, eq. 1):

    penalty(T_hat, T) = 0   if T_hat == T  (token sets equal)
                       0.5  if T_hat ⊂ T   (proper subset)
                       1    if T_hat ∩ T = ∅
                       0.75 otherwise (partial overlap, neither equal nor subset)

The paper defines the first three cases; we add a "partial overlap"
penalty halfway between the subset and disjoint cases for predictions that
overlap with gold but are neither equal nor a subset, so we get a complete
function over all possible token sets. We log this assumption.

For each test ingredient line we:
  1. Run the trained NER → list of predicted entity strings.
  2. Read gold entity strings from the annotation.
  3. Greedy-match each predicted entity to the gold entity it overlaps with
     most (Jaccard over word tokens). Compute the per-pair penalty.
  4. Unmatched predicted entities and unmatched gold entities each contribute
     penalty 1 (false positive / false negative).
  5. Per-line score = mean of all per-pair / unmatched penalties (skipped if
     line has no gold entities AND no predictions).

We report mean penalty across all 424 test lines. Lower is better.
"""

import argparse
import json
import sys
from pathlib import Path

import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging_setup import setup_logging  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NER_DIR = PROJECT_ROOT / "ner"
DATASET_DIR = NER_DIR / "dataset"
DEFAULT_TEST_FILES = [DATASET_DIR / "test_0.json", DATASET_DIR / "test_1.json"]
DEFAULT_MODEL_DIR = NER_DIR / "model_v3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_files", nargs="+", type=Path, default=DEFAULT_TEST_FILES)
    p.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--label", default="food")
    p.add_argument("--results_path", type=Path, default=NER_DIR / "ner_results.json")
    return p.parse_args()


def tokens(s: str) -> set:
    return {t for t in s.lower().split() if t}


def pair_penalty(pred_tokens: set, gold_tokens: set) -> float:
    if pred_tokens == gold_tokens:
        return 0.0
    if pred_tokens and pred_tokens.issubset(gold_tokens):
        return 0.5
    if not pred_tokens.intersection(gold_tokens):
        return 1.0
    return 0.75


def evaluate_line(nlp, content: str, gold_spans, label: str):
    doc = nlp(content)
    pred_strs = [ent.text for ent in doc.ents if ent.label_ == label]
    gold_strs = [content[s:e] for s, e, lab in gold_spans if lab == label]

    pred_sets = [tokens(s) for s in pred_strs]
    gold_sets = [tokens(s) for s in gold_strs]

    matched_gold = set()
    penalties = []
    for ps in pred_sets:
        if not gold_sets:
            penalties.append(1.0)
            continue
        best_idx, best_score = -1, -1.0
        for j, gs in enumerate(gold_sets):
            if j in matched_gold:
                continue
            inter = len(ps.intersection(gs))
            union = len(ps.union(gs)) or 1
            jac = inter / union
            if jac > best_score:
                best_score, best_idx = jac, j
        if best_idx == -1:
            penalties.append(1.0)
        else:
            matched_gold.add(best_idx)
            penalties.append(pair_penalty(ps, gold_sets[best_idx]))

    for j in range(len(gold_sets)):
        if j not in matched_gold:
            penalties.append(1.0)

    return penalties, pred_strs, gold_strs


def main() -> int:
    args = parse_args()
    log = setup_logging("eval_ner")
    log.info("Args: %s", vars(args))

    nlp = spacy.load(args.model_dir)
    log.info("Loaded NER pipeline from %s", args.model_dir)

    rows = []
    for f in args.test_files:
        with open(f, "r", encoding="utf-8") as fh:
            rows.extend(json.load(fh))
    log.info("Loaded %d test lines from %d files", len(rows), len(args.test_files))

    all_penalties = []
    line_means = []
    n_lines_scored = 0
    breakdown = {"exact": 0, "subset": 0, "partial": 0, "disjoint": 0}
    for r in rows:
        pens, _, _ = evaluate_line(nlp, r["content"], r.get("entities", []), args.label)
        if not pens:
            continue
        n_lines_scored += 1
        all_penalties.extend(pens)
        line_means.append(sum(pens) / len(pens))
        for p in pens:
            if p == 0.0:
                breakdown["exact"] += 1
            elif p == 0.5:
                breakdown["subset"] += 1
            elif p == 0.75:
                breakdown["partial"] += 1
            else:
                breakdown["disjoint"] += 1

    mean_penalty = sum(all_penalties) / max(1, len(all_penalties))
    line_mean_penalty = sum(line_means) / max(1, len(line_means))

    results = {
        "n_test_lines": len(rows),
        "n_lines_scored": n_lines_scored,
        "n_pair_penalties": len(all_penalties),
        "mean_pair_penalty": mean_penalty,
        "mean_line_penalty": line_mean_penalty,
        "breakdown": breakdown,
    }
    log.info("Results: %s", results)
    args.results_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote results to %s", args.results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
