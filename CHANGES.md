# What changed, mapped to the review (AI-PRISM_Review.pdf)

Encoder: Clinical ModernBERT, windowed. Default overlap 10, max_chunks 24 (raise if reports are truncated). Switching to a
512-token encoder (e.g. Bio_ClinicalBERT) only needs `model_name`, `max_chunks: 5`.

| Issue | Fix | Where |
|---|---|---|
| 2 Encoder reads ~126 tokens | Clinical ModernBERT kept. Reports are split into overlapping windows that fit the trained positions (auto-detected from the weights: 127 slots -> 125 text tokens + CLS/SEP), encoded, and pooled (mean / max / attention). `window_tokens` larger than the trusted length is refused. Startup prints the real class and trusted positions. | `src/chunking.py`, `src/encoder_check.py`, `src/model.py`, `tools/check_encoder.py` |
| 3 Unknown treated as 0 | Labels are 1 / 0 / NaN. Masked BCE (unknown = no gradient), masked metrics, NaN-aware prevalence for bias init and pos_weight, unknown indicators in stratification. Patients with no known label are excluded, not zeroed. MAF + sequenced-ID mode builds real labels. | `src/data.py`, `src/losses.py`, `src/metrics.py` |
| 4 Loss rewards "mutated" for all | ASL removed. Plain masked BCE (optional pos_weight). Training AUROC logged each epoch next to loss and the no-information loss floor. | `src/losses.py`, `src/train.py` |
| 1 Panel checks | `mutation_definition` (nonsilent / kras_hotspot), TMB from MAF or file, TMB-only baseline and text+TMB vs TMB-only AUROC difference with bootstrap CI, per-gene known positive / negative counts with warnings. | `src/data.py`, `src/baselines.py` |
| 5 Not reproducible | Committed config per experiment; unknown config keys are errors; each run saves `config_used.json`, `config_source.yaml`, git commit + library versions; `.gitignore` no longer hides run outputs; stale `hyperparameters.pdf` removed, table generated from config; README rewritten to match the code. | `configs/`, `src/config.py`, `src/pipeline.py`, `tools/make_hyperparameter_table.py` |
| 6 Site confound | Site code parsed; site-only baseline; `cv_scheme: site_grouped` (StratifiedGroupKFold); optional unsupervised boilerplate stripping per site. | `src/data.py`, `src/baselines.py`, `src/preprocess.py` |
| 7 Reports state the mutation | `gene_mentions.csv` lists every report naming a panel gene with context; `mask_genes` replaces gene names and variant notation; baselines run `raw` and `masked` variants. p53 immunostain is left unmasked by default (`mask_p53_ihc`). | `src/preprocess.py`, `src/pipeline.py` |
| 8 ~25-patient model selection, ~8 updates | Validation split, best-epoch selection and F1 threshold tuning removed. Fixed schedule; batch of 8 patients (windows are short); linear heads and a frozen encoder by default; repeated CV; headline = pooled out-of-fold AUROC/AUPRC with bootstrap CI. | `src/train.py`, `configs/` |
| 9 No baselines first | `src.embed` caches frozen vectors; `src.baselines` runs prevalence, site, TMB, TF-IDF, frozen-embedding (+TMB) with label-permutation p-values and a Bonferroni flag. | `src/embed.py`, `src/baselines.py` |
| 10 Minor | Linear heads (specification) with `mlp` optional; last accumulation group divided by its true size; no phase-2 restart from a checkpoint (fixed schedule); Mut_Genes tokens validated (an entry like "KRAS G12C" raises); text cleaning and truncation count reported. | various |

Not done (needs data or decisions that are not in the repository): rebuilding labels from a MAF, TMB values, the
hand check of the 17 gene-naming reports, and running anything on the real reports. See the notes in the reply.
