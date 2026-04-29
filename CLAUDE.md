# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project goal

Replicate the metrics reported in *RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation* (Bień et al., INLG 2020) — https://aclanthology.org/2020.inlg-1.4.pdf — **without using the published `mbien/recipenlg` HuggingFace checkpoint**. We retrain everything from scratch.

### Replication targets

| Metric | Paper value (RecipeNLG) | Source |
|---|---|---|
| Cosine similarity (TF-IDF, mean of max-over-10) | 0.666 | §5.3 |
| LanguageCheck errors / recipe (mean) | 2.78 (gen), 3.64 (gold) | §5.3 |
| BLEU (sentence, 10 gens as refs, gold as hyp, smoothing4) | 0.866 | Table 2 |
| GLEU | 0.662 | Table 2 |
| WER (mean of min-of-10) | 0.751 | Table 2 |
| NER penalty (custom, no value reported in paper) | report on 424-line test set | §5.1 |

Scope (per user):
- Train **only** on full RecipeNLG (~2.23M); skip the Recipe1M+ baseline row of Table 2.
- Modernized stack (transformers 5.x, language_tool_python, spaCy 3.x).
- Full paper-scale GPT-2 fine-tuning.

## Data on disk

- `/mmfs1/projects/changhui.yan/m.afzalsiddique/datasets/RecipeNLG/full_dataset.csv` — 2,231,142 rows; columns `title, ingredients, directions, link, source, NER`. The `NER` column already contains paper-§5.1-style entity lists, so we do NOT re-run NER over 2.2M recipes for the generation task.
- `ner/dataset/{train,test}_{0,1}.json` — manually annotated single-line ingredients with `food` spans. Train: 1690, Test: 424.
- `ner/model/` — old spaCy 2.x pretrained model (UNUSED — we retrain into `ner/model_v3/`).

## Cluster conventions

- PBS cluster, no GPU on login node. GPU queues: `gpus`, `preemptible` (with `glist` tag for specific GPU types like `l40s`, `a100`).
- **Do not pin `glist=a100`** by default — request a generic `ngpus=1` so jobs can land on any free GPU; user has been overriding to `glist=l40s` on `preemptible` for short jobs.
- **Every runnable script — including smoke tests — gets its own PBS submission script.** Nothing runs on the login node.
- `HF_HOME=/mmfs1/scratch/m.afzalsiddique/hfcache`.
- All HPs and paths defined explicitly at the top of each script via argparse (no buried magic numbers; even library-default values are listed).
- Logging: every entry point calls `utils.logging_setup.setup_logging(name)` first; logs go to `logs/{YYYYmmdd-HHMMSS}_{script}.log` and stdout.

## Repo layout

```
recipenlg/
├── utils/
│   └── logging_setup.py     # shared timestamped logger
├── ner/
│   ├── dataset/             # train_*.json, test_*.json (provided)
│   ├── model/               # spaCy v2 model from upstream — unused
│   ├── train_ner.py         # spaCy v3 trainer (DocBin from JSON)
│   └── eval_ner.py          # paper's penalty(T_hat, T) metric
├── generation/
│   ├── prepare_text.py      # CSV -> tagged plain-text + gold_test_100.json
│   ├── tokenization.py      # plain-text -> packed unsupervised.h5
│   ├── run_lm_finetuning_new.py  # GPT-2 finetune (Trainer 5.x)
│   ├── run_generation.py    # model.generate over gold_test_100.json
│   └── data/                # outputs land here (unsupervised_train.txt, etc.)
├── eval/
│   ├── cosine_similarity.py # TF-IDF helper (re-implemented from scratch)
│   ├── replicate_metrics.py # cosine + LanguageCheck + BLEU + GLEU + WER
│   └── evaluation.ipynb     # original notebook (kept for reference)
├── pbs/
│   ├── env.sh               # sourced by every job; activates .venv
│   ├── setup_venv.pbs       # bootstrap project-local venv (idempotent)
│   ├── smoke.pbs            # end-to-end pipeline on 5k rows / 100 train steps
│   ├── train_ner.pbs / eval_ner.pbs
│   ├── prepare_text.pbs / tokenize.pbs
│   ├── train_lm.pbs         # full GPT-2 fine-tuning, GPU, up to 168h
│   ├── generate.pbs         # 1000 generations from gold_test_100.json
│   └── eval.pbs             # final metric computation
├── logs/                    # auto-created; timestamped per-run logs
├── requirements.txt         # modernized
└── CLAUDE.md                # this file
```

Original 2020-era files kept untouched (do NOT rely on these to run):
`generation/preparation.py`, `generation/run_lm_finetuning.py`, `generation/run_lm_finetuning_tpu.py`, `ner/NER.ipynb`, `ner/Language2_0.ipynb`, `generation/dataset2text.ipynb`, `eval/evaluation.ipynb`, `scraping-scripts/`, `recipes_spider/`.

## End-to-end pipeline (qsub order)

1. `qsub pbs/setup_venv.pbs` — one-time, creates `.venv/` with modern stack.
2. `qsub pbs/smoke.pbs` — sanity-check on 5k rows / 100 train steps. **Done; passed.**
3. NER track (CPU, fast):
   - `qsub pbs/train_ner.pbs` — saves to `ner/model_v3/`.
   - `qsub pbs/eval_ner.pbs` — writes `ner/ner_results.json`.
4. Generation track:
   - `qsub pbs/prepare_text.pbs` — writes `generation/data/{unsupervised_train.txt, unsupervised_test.txt, gold_test_100.json}`.
   - `qsub pbs/tokenize.pbs` — writes `generation/data/unsupervised.h5` (depends on prepare_text).
   - `qsub pbs/train_lm.pbs` — fine-tune GPT-2; up to 168h on 1 GPU; saves to `generation/model_out/` (depends on tokenize).
   - `qsub pbs/generate.pbs` — produces `generation/data/generated_recipenlg.json` (1000 strings) (depends on train_lm).
   - `qsub pbs/eval.pbs` — produces `eval/results.json` (depends on generate).

## Current state (2026-04-29)

- Setup: `pbs/setup_venv.pbs` succeeded; `.venv` populated with transformers 5.7.0, torch (CUDA 12.1), spaCy 3.x, language_tool_python, etc.
- Smoke test: `pbs/smoke.pbs` ran end-to-end successfully after a transformers-5.x compatibility patch in `generation/run_lm_finetuning_new.py` (see "Known fixes" below).
- **Currently running** (per user, 2026-04-29):
  - `pbs/train_ner.pbs` (queue: preemptible)
  - `pbs/prepare_text.pbs` (queue: preemptible)
- Not yet launched: `pbs/tokenize.pbs`, `pbs/train_lm.pbs`, `pbs/generate.pbs`, `pbs/eval.pbs`, `pbs/eval_ner.pbs`.

## Known fixes / compatibility notes

- transformers 5.7 dropped `TrainingArguments(overwrite_output_dir=...)` and renamed `Trainer(tokenizer=...)` → `processing_class=...`. `run_lm_finetuning_new.py` feature-detects via `inspect.signature` and works on both 4.x and 5.x.
- `language_tool_python` requires Java 11+ at runtime; first invocation downloads the LanguageTool jar to `~/.cache/language_tool_python/`.
- `language_check` (the original 2020 dependency) is unmaintained and incompatible with modern Python/Java — replaced wholesale by `language_tool_python`. The `ruleId` filter `WHITESPACE_RULE` is still the right thing to skip (matches `eval/evaluation.ipynb`).
- The old `cosine_similarity` module imported by `eval/evaluation.ipynb` is missing from the upstream repo; reimplemented at `eval/cosine_similarity.py` using `sklearn.feature_extraction.text.TfidfVectorizer`.

## How to add new pipeline steps

1. Add the Python script under the relevant package (`generation/`, `ner/`, `eval/`, etc.).
2. Top of file: argparse with explicit defaults for **every** HP (no magic numbers, even when matching library defaults).
3. First call in `main()`: `setup_logging("script_name")`.
4. Add a matching `pbs/<name>.pbs` that:
   - Sources `pbs/env.sh`.
   - Requests `ngpus=1` without `glist=` unless there's a specific reason.
   - `cd "$PBS_O_WORKDIR"` first, `set -euo pipefail` if multi-step.
   - Mirrors stdout/stderr to `logs/` via `#PBS -j oe` + `#PBS -o logs/`.
5. Update this CLAUDE.md "End-to-end pipeline" section with the new step.
