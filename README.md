# User Guide

A reproducible pipeline for fine-tuning GPT-2 on the [RecipeNLG](https://recipenlg.cs.put.poznan.pl/) corpus and evaluating recipe generation, paired with an autonomous hyperparameter-search component (`autoresearch-master/`) that runs short training experiments on a RecipeNLG subset and ports the winners back into the main fine-tune.

The current best fine-tuned model achieves `eval_loss=1.245` against the original mbien/recipenlg HuggingFace baseline. Hyperparameters from the most recent autoresearch winners (LR `1e-4`, β₁ `0.8`, weight decay `0.1`) are already applied in `pbs/train_lm.pbs`.

## Repository layout

```
recipenlg/
├── autoresearch-master/      # 5-minute HP-search loop (Karpathy-lineage trainer + Muon)
│   ├── train.py              # editable trainer (architecture + optimizer)
│   ├── prepare.py            # data prep + eval harness (read-only)
│   ├── prepare_job.pbs       # one-time data preparation
│   ├── train_job.pbs         # 5-min training run on A10
│   ├── program.md            # experiment loop policy
│   ├── results.tsv           # append-only per-experiment log
│   └── CLAUDE.md             # full HP mapping autoresearch ↔ fine-tune
├── generation/               # GPT-2 fine-tune + recipe generation
│   ├── prepare_text.py       # CSV → train/test/gold text splits
│   ├── tokenization.py       # text → tokenized HDF5 shards
│   ├── run_lm_finetuning_new.py   # training entry point (HF Trainer)
│   ├── run_generation.py     # sampling entry point
│   ├── data/                 # prepared text + tokenized data + generations
│   └── model_out/            # trained checkpoints (created by train_lm.pbs)
├── ner/                      # spaCy NER for ingredient extraction (optional)
├── eval/
│   ├── replicate_metrics.py  # BLEU / GLEU / WER / cosine / language_check
│   └── results.json          # populated by pbs/eval.pbs
├── pbs/                      # PBS submission scripts (see Pipeline section)
├── utils/                    # logging helpers
├── requirements.txt
└── CLAUDE.md                 # project context and HP transfer rules
```

## Prerequisites

- HPC cluster with the **PBS scheduler** (`qsub`, `qstat`).
- **GPUs**: A10 for autoresearch, A100 (preferred) or A40/L40S for the full fine-tune.
- **Python** 3.10+.
- **`uv`** package manager — only required for the autoresearch component.
- **RecipeNLG dataset CSV** — download `full_dataset.csv` from <https://recipenlg.cs.put.poznan.pl/>. Set its absolute path in your shell:

  ```bash
  export RECIPENLG_CSV=/path/to/full_dataset.csv
  ```

  Then edit `pbs/prepare_text.pbs` and `autoresearch-master/prepare_job.pbs` to point at `$RECIPENLG_CSV` (one line each), or pass the path as an argument inside those scripts.
- **Site-specific PBS directives** — these scripts intentionally do **not** set `#PBS -W group_list=...` or `#PBS -M ...`. Add lines for your site/account/email if your scheduler requires them.

## One-time setup

```bash
# Main fine-tune environment (creates .venv with PyTorch CUDA 12.1 + requirements.txt)
qsub pbs/setup_venv.pbs

# Autoresearch environment (uv-managed)
cd autoresearch-master
uv sync
cd ..
```

After both finish, `pbs/env.sh` will activate `.venv/` automatically for every PBS job in this repo. The autoresearch sub-tree uses `uv run` and self-resolves.

---

## Reproducing the autoresearch HP search

The autoresearch component runs a single 5-minute training experiment per submission and appends `val_bpb` to `results.tsv`. Lower is better. Use this to validate hyperparameters before transferring them to the full fine-tune.

```bash
cd autoresearch-master

# Step A — one-time data preparation (cached under ~/.cache/autoresearch/)
qsub prepare_job.pbs

# Step B — single 5-min training run on A10
qsub train_job.pbs

# Step C — inspect results
grep "^val_bpb:\|^peak_vram_mb:" run.log
tail results.tsv
```

The experiment loop policy and HP-knob inventory live in `autoresearch-master/program.md`. The full transfer mapping back to the GPT-2 fine-tune (which knobs map directly, which need translation, which are non-transferable) lives in `autoresearch-master/CLAUDE.md`.

Current best on this branch: commit `a046891`, `val_bpb=0.6087` (re-runs land near `0.6091` at the on-node noise floor).

---

## Reproducing the full GPT-2 fine-tune

Five PBS stages, chained via `afterok` dependencies so the next stage starts only after the previous succeeds.

### Stage 1 — Prepare text

Splits the CSV into 95/5 train/test text files and a gold subset for evaluation.

```bash
JOB_PREP=$(qsub pbs/prepare_text.pbs)
```

Outputs: `generation/data/unsupervised_train.txt`, `unsupervised_test.txt`, `gold_test_100.json`.

### Stage 2 — Tokenize

Tokenizes with the GPT-2 BPE into a single HDF5 shard at `block_size=1024`.

```bash
JOB_TOK=$(qsub -W depend=afterok:$JOB_PREP pbs/tokenize.pbs)
```

Output: `generation/data/unsupervised.h5`.

### Stage 3 — Fine-tune GPT-2

Two epochs over the full corpus on a single A100. Walltime budgeted at 168 h.

```bash
JOB_TRAIN=$(qsub -W depend=afterok:$JOB_TOK pbs/train_lm.pbs)
```

Output: `generation/model_out/` (HF checkpoint with config, tokenizer, weights).

The CLI inside `pbs/train_lm.pbs` already encodes the autoresearch-derived HPs:

| Arg | Value | Source |
|---|---|---|
| `--learning_rate` | `1e-4` | autoresearch exp #15 (UNEMBEDDING_LR 2× win) |
| `--adam_beta1` | `0.8` | autoresearch exp #4 (β₁=0.8 outperforms 0.9) |
| `--adam_beta2` | `0.95` | autoresearch baseline |
| `--adam_epsilon` | `1e-8` | conservative for pretrained AdamW |
| `--weight_decay` | `0.1` | autoresearch high-WD direction (mapped) |
| `--warmup_ratio` | `0.0` | autoresearch baseline |
| `--label_smoothing_factor` | `0.0` | autoresearch baseline |
| `--max_grad_norm` | `1.0` | fine-tune safety |
| `--lr_scheduler_type` | `linear` | matches autoresearch warmdown |
| `--attn/embd/resid_pdrop` | `0.1` | GPT-2 default |

### Stage 4 — Generate

Samples 10 candidate recipes per gold prompt with nucleus sampling.

```bash
JOB_GEN=$(qsub -W depend=afterok:$JOB_TRAIN pbs/generate.pbs)
```

Output: `generation/data/generated_recipenlg.json`.

### Stage 5 — Evaluate

Computes the five RecipeNLG paper metrics: cosine similarity (TF-IDF), sentence BLEU (smoothing method 4), GLEU, WER, and LanguageTool error rate.

```bash
qsub -W depend=afterok:$JOB_GEN pbs/eval.pbs
```

Output: `eval/results.json`.

### Smoke test 

End-to-end pipeline on 5k rows / 100 steps in ~2 hours on an L40S:

```bash
qsub pbs/smoke.pbs
```

### Optional NER side-track

```bash
qsub pbs/train_ner.pbs
qsub pbs/eval_ner.pbs
```

---

## Hyperparameter transfer rules (autoresearch → fine-tune)

Quick reference for porting a new autoresearch winner into `pbs/train_lm.pbs`:

| autoresearch finding | fine-tune action |
|---|---|
| `UNEMBEDDING_LR` 2× wins | scale `--learning_rate` by the same factor (cap at 2× current) |
| `ADAM_BETAS` (b1, b2) | set `--adam_beta1 b1 --adam_beta2 b2` (validate on smoke) |
| `WEIGHT_DECAY` direction (high vs low) | high → `0.1`; low → `0.01`. **Do not copy magnitude** — Muon vs AdamW |
| `WARMUP_RATIO` / `WARMDOWN_RATIO` | `--warmup_ratio` directly; keep `--lr_scheduler_type linear` |
| `LABEL_SMOOTHING` | `--label_smoothing_factor` directly |
| `GRADIENT_CLIP_NORM` (Muon) | keep `--max_grad_norm 1.0` for fine-tune safety |
| `MATRIX_LR`, `EMBEDDING_LR` | not transferred (Muon-only mechanism) |

Full mapping with rationale: see `CLAUDE.md` and `autoresearch-master/CLAUDE.md`.

---

## Outputs and artifacts

| Stage | Path |
|---|---|
| Prepared text | `generation/data/unsupervised_{train,test}.txt`, `gold_test_100.json` |
| Tokenized data | `generation/data/unsupervised.h5` |
| Fine-tuned model | `generation/model_out/` |
| Generations | `generation/data/generated_recipenlg.json` |
| Eval metrics | `eval/results.json` |
| Per-job logs | `logs/<timestamp>_<script>.log` |
| Autoresearch run log | `autoresearch-master/run.log` |
| Autoresearch experiment log | `autoresearch-master/results.tsv` |

---

## Monitoring and troubleshooting

```bash
# Job status
qstat -u $USER
qstat <JOBID>

# Live training loss / eval_loss
grep -E "loss|eval_loss|perplexity" logs/*train_lm*.log | tail

# Generation sanity check
ls -la generation/data/generated_recipenlg.json

# Eval results
cat eval/results.json
```

Common issues:

- **Job sits in queue (`Q`)** — requested GPU type has no free nodes. Check availability and either wait or edit the `glist=` directive in the PBS script (e.g. `a100` → `a40`).
- **`Dataset CSV not found`** — `RECIPENLG_CSV` env var unset, or the path inside `pbs/prepare_text.pbs` / `autoresearch-master/prepare_job.pbs` is stale.
- **HuggingFace cache misses** — `pbs/env.sh` sets `HF_HOME` to a scratch path. Make sure that path is writable from the compute node.
- **Java not found in eval** — `eval/replicate_metrics.py` uses `language_tool_python`, which downloads a LanguageTool jar on first use and needs `JAVA_HOME` to point at a Java 11+ runtime. The `eval.pbs` script prints `java -version` for diagnostics.
- **`eval_loss` regresses vs 1.245** — revert the most recent HP changes in `pbs/train_lm.pbs` in this order: `--weight_decay` → `--adam_beta1` → `--learning_rate` (drop to `7.5e-5`).

---

## Reference

- Paper: Bień et al., *RecipeNLG: A Cookbook to Cooking with NLG*, INLG 2020.
- Dataset: <https://recipenlg.cs.put.poznan.pl/>
- HuggingFace baseline: [`mbien/recipenlg`](https://huggingface.co/mbien/recipenlg) (`eval_loss=1.245`, `perplexity=3.47`).
