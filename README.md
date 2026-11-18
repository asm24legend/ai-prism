# AI-PRISM (LUAD): Clinical ModernBERT (windowed) with per-gene heads

Predicts mutation status of **KRAS, MUC16, RYR2, TP53, TTN** from lung adenocarcinoma pathology reports.

```
report text -> clean / (optional) mask gene names
            -> overlapping windows (overlap 10, up to 24) sized to the encoder's trained positions
            -> Simonlee711/Clinical_ModernBERT -> window vectors -> pool (mean | max | attention) -> report vector z
            -> one linear head per gene -> sigmoid -> p_gene
labels: 1 mutated / 0 sequenced, not mutated / NaN unknown  -> masked loss, masked metrics
```

Why windows: the published `Simonlee711/Clinical_ModernBERT` has been reported (review, Issue 2) to load as a classic
BERT with only ~127 trained position slots. The code reads the loaded weights, sizes windows to the trained positions
(~125 text tokens + CLS/SEP) and refuses larger `window_tokens`. If the checkpoint turns out to be a true ModernBERT
(rotary), the window widens automatically (capped at 1,024). Run `tools/check_encoder.py` first. Any other encoder
(e.g. `emilyalsentzer/Bio_ClinicalBERT`, 512 tokens) can be swapped in via `model_name`.

## Layout
| Path | Purpose |
|---|---|
| `configs/*.yaml` | one committed config per experiment (never edit after a run) |
| `src/config.py` | defaults + validation (unknown keys are an error) |
| `src/data.py` | three-state labels (table or MAF), site codes, patient-level / site-grouped splits |
| `src/preprocess.py` | cleaning, gene-name masking, optional site-boilerplate stripping, gene-mention report |
| `src/chunking.py`, `src/encoder_check.py` | windowing; position-table check; masked-word test |
| `src/model.py`, `src/losses.py`, `src/metrics.py` | model, masked BCE, masked metrics + bootstrap CIs |
| `src/embed.py` | encode every report once with the frozen encoder, cache window vectors |
| `src/baselines.py` | prevalence / site / TMB / TF-IDF / frozen-embedding baselines with permutation tests |
| `src/train.py` | fine-tuning (only if a baseline shows signal) |
| `tools/check_encoder.py` | run on any encoder before use |
| `tests/` | `pytest` unit tests |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # GPU: install the CUDA build of torch first
pytest -q
```

## Order of work (follows the review)
```bash
python tools/check_encoder.py --model Simonlee711/Clinical_ModernBERT     # 1. encoder sanity
python -m src.embed     --config configs/frozen_baselines.yaml            # 2. cache frozen window vectors (raw + masked)
python -m src.baselines --config configs/frozen_baselines.yaml            # 3. baselines, site-grouped repeated CV
python -m src.train     --config configs/finetune.yaml                    # 4. only if step 3 shows signal
```

## Labels - read this
The label table lists only genes found mutated, so on its own it cannot say which genes were *tested*.
Without a MAF the code treats a listed patient's absent gene as 0 (whole-exome assumption, warned at start-up) and a
patient with an empty record as **unknown**. For real three-state labels and TMB set in the config:
`maf_path`, `sequenced_ids_path`, `mutation_definition` (`nonsilent` or `kras_hotspot`).

## Outputs (`runs/<exp>/`)
`config_used.json`, `config_source.yaml`, `run_info.json` (git commit), `label_counts.csv`, `gene_mentions.csv`,
`summary.json` (pooled out-of-fold AUROC/AUPRC + bootstrap CI), `oof_predictions.csv`, `baseline_results.csv`,
`text_plus_tmb_vs_tmb.csv`. Copy `runs/<exp>/` into version control with the config (the repo `.gitignore` no longer
hides `runs/*/summary.json` etc.).

Hyperparameter tables: `python tools/make_hyperparameter_table.py configs/finetune.yaml` (generated, not hand-written).
