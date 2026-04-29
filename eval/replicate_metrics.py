"""
Replicate the five generation metrics from RecipeNLG (Bień et al., INLG 2020)
§5.3 / Table 2 for the RecipeNLG row.

Inputs:
  --gold_json   gold_test_100.json  (output of generation/prepare_text.py)
  --generated_json  generated_recipenlg.json (output of run_generation.py;
                    flat list of N*num_return_sequences strings, gold-major)

Aggregation matches eval/evaluation.ipynb:
  cosine: mean over k of max_i cosine_sim([gold[k], gen[k*K + i]])[0,1]
  language: mean over generated of len(matches except WHITESPACE_RULE);
            also computed over gold for the §5.3 baseline comparison
  BLEU:    mean over k of sentence_bleu(refs=gen[k*K:(k+1)*K], hyp=gold[k],
                                        smoothing_function=method4)
  GLEU:    mean over k of sentence_gleu(refs, hyp)
  WER:     mean over k of min_i jiwer.wer(gen[k*K + i], gold[k])

All HPs / paths are listed explicitly via argparse below.
"""

import argparse
import json
import sys
from pathlib import Path

import nltk
import numpy as np
from jiwer import wer as jiwer_wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging_setup import setup_logging  # noqa: E402
from eval.cosine_similarity import cosine_sim  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold_json", type=Path,
                   default=PROJECT_ROOT / "generation" / "data" / "gold_test_100.json")
    p.add_argument("--generated_json", type=Path,
                   default=PROJECT_ROOT / "generation" / "data" / "generated_recipenlg.json")
    p.add_argument("--output_json", type=Path,
                   default=PROJECT_ROOT / "eval" / "results.json")
    p.add_argument("--num_return_sequences", type=int, default=10,
                   help="Number of generations per gold prompt (must match generation step).")
    p.add_argument("--language_check_rule_skip", default="WHITESPACE_RULE",
                   help="LanguageTool rule_id to ignore (matches the eval notebook).")
    p.add_argument("--language_tool_lang", default="en-US")
    p.add_argument("--skip_language_check", action="store_true",
                   help="Skip LanguageTool (slow / requires Java).")
    return p.parse_args()


def load_gold_texts(path: Path):
    rows = json.loads(path.read_text())
    return [r["full_text"] for r in rows]


def load_generated(path: Path):
    return json.loads(path.read_text())


def cosine_metric(gold, generated, k):
    bests = []
    for idx, g in enumerate(tqdm(gold, desc="cosine")):
        best = 0.0
        for i in range(k):
            cand = generated[idx * k + i]
            sim = cosine_sim([g, cand])[0, 1]
            if sim > best:
                best = float(sim)
        bests.append(best)
    return float(np.mean(bests)), bests


def language_check_metric(texts, tool, skip_rule):
    counts = []
    for t in tqdm(texts, desc="language_tool"):
        matches = tool.check(t)
        counts.append(sum(1 for m in matches if m.ruleId != skip_rule))
    return float(np.mean(counts)), counts


def bleu_metric(gold, generated, k):
    smoothie = SmoothingFunction().method4
    scores = []
    for idx, g in enumerate(tqdm(gold, desc="bleu")):
        refs = generated[idx * k : (idx + 1) * k]
        scores.append(sentence_bleu(refs, g, smoothing_function=smoothie))
    return float(np.mean(scores)), scores


def gleu_metric(gold, generated, k):
    scores = []
    for idx, g in enumerate(tqdm(gold, desc="gleu")):
        refs = generated[idx * k : (idx + 1) * k]
        scores.append(sentence_gleu(refs, g))
    return float(np.mean(scores)), scores


def wer_metric(gold, generated, k):
    scores = []
    for idx, g in enumerate(tqdm(gold, desc="wer")):
        best = float("inf")
        for i in range(k):
            cand = generated[idx * k + i]
            try:
                w = jiwer_wer(g, cand)
            except ValueError:
                w = 1.0
            if w < best:
                best = w
        scores.append(best)
    return float(np.mean(scores)), scores


def main() -> int:
    args = parse_args()
    log = setup_logging("replicate_metrics")
    log.info("Args: %s", vars(args))

    nltk.download("punkt", quiet=True)
    gold = load_gold_texts(args.gold_json)
    generated = load_generated(args.generated_json)
    K = args.num_return_sequences
    log.info("Loaded gold=%d generated=%d K=%d", len(gold), len(generated), K)
    if len(generated) != len(gold) * K:
        log.warning("len(generated)=%d but len(gold)*K=%d — eval will fail or be partial.",
                    len(generated), len(gold) * K)

    results = {}

    cos_mean, _ = cosine_metric(gold, generated, K)
    results["cosine_similarity_mean"] = cos_mean
    log.info("Cosine similarity (mean of max-of-K): %.4f  (paper: 0.666)", cos_mean)

    if not args.skip_language_check:
        try:
            import language_tool_python
            tool = language_tool_python.LanguageTool(args.language_tool_lang)
            gen_lc_mean, _ = language_check_metric(generated, tool, args.language_check_rule_skip)
            results["language_check_errors_generated_mean"] = gen_lc_mean
            log.info("LanguageCheck errors (generated): %.3f  (paper: 2.78)", gen_lc_mean)

            gold_lc_mean, _ = language_check_metric(gold, tool, args.language_check_rule_skip)
            results["language_check_errors_gold_mean"] = gold_lc_mean
            log.info("LanguageCheck errors (gold): %.3f  (paper: 3.64)", gold_lc_mean)
            tool.close()
        except Exception as e:
            log.error("LanguageCheck failed: %s", e)
            results["language_check_error"] = str(e)
    else:
        log.info("Skipping LanguageCheck (per --skip_language_check).")

    bleu_mean, _ = bleu_metric(gold, generated, K)
    results["bleu_mean"] = bleu_mean
    log.info("BLEU (sentence, smoothing4): %.4f  (paper: 0.866)", bleu_mean)

    gleu_mean, _ = gleu_metric(gold, generated, K)
    results["gleu_mean"] = gleu_mean
    log.info("GLEU: %.4f  (paper: 0.662)", gleu_mean)

    wer_mean, _ = wer_metric(gold, generated, K)
    results["wer_mean"] = wer_mean
    log.info("WER (mean of min-of-K): %.4f  (paper: 0.751)", wer_mean)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    log.info("Wrote results to %s", args.output_json)
    log.info("RESULTS: %s", json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
