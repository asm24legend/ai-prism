# AI-PRISM (LUAD): Clinical ModernBERT with per-gene heads

Predicts mutation status of **KRAS, MUC16, RYR2, TP53, TTN** from lung adenocarcinoma pathology reports.

```
report text (as given) -> Clinical ModernBERT (up to 8192 tokens) -> mean-pooled embedding z (768-d)
            -> 5 separate MLPs (768 -> 256 -> 128 -> 1) -> sigmoid -> p_gene
```

## Layout
| Path | Purpose |
|---|---|
| `config.yaml` | all settings (paths, model, loss, learning rates, phases, folds) |
| `data/` | `PRISM_LUAD_Dataset_-_Sheet1.csv` (labels) and `TCGA_Reports.xlsx` (report text, matched by file name) |
| `src/data.py` | label parsing, text join, Dataset, patient-level multilabel-stratified splits |
| `src/preprocess.py` | optional cleanup and gene-name masking (both off by default; raw text is used) |
| `src/model.py` | Clinical ModernBERT + one MLP per gene, freezing helper |
| `src/losses.py` | BCE (optional pos_weight) and asymmetric loss |
| `src/train.py` | two-phase training, 5-fold CV, metrics, outputs |
| `runs/` | results per experiment (created at run time) |

## Setup (VS Code terminal)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # on a GPU machine install the CUDA build of torch first (pytorch.org)
```
In VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> pick `.venv`.

## Run (from the project root)
```bash
python -m src.train --config config.yaml
python -m src.train --config config.yaml --override n_folds=1 max_len=2048 output_dir=runs/quick
```
Needs a GPU (about 16 GB for `max_len=4096`; set `grad_ckpt=true` or lower `max_len` if memory runs out).
The first run downloads `Simonlee711/Clinical_ModernBERT` from Hugging Face.

## Training phases
1. **Phase 1** (`phase1_epochs`): encoder fully frozen, only the 5 gene heads train (`lr_head`).
2. **Phase 2** (`phase2_epochs`): top `unfreeze_layers` encoder layers unfrozen at the lower `lr_encoder`.

Each gene's output bias starts at the log-odds of its training prevalence. The best checkpoint is chosen by
validation macro-AUPRC; per-gene F1 thresholds are tuned on validation only.

## Outputs (`runs/<exp>/`)
`summary.json` (mean/std over folds, per-gene AUROC/AUPRC), `oof_predictions.csv`, `fold*/metrics.json`, `config_used.json`.

## Notes
- 191 LUAD patients; 183 have a report in `TCGA_Reports.xlsx` (8 are dropped and listed at start-up).
- Report texts are used directly, with no OCR repair, section reordering or tokenizer vocabulary extension.
- 17 of the 183 reports name one of the five genes. Set `mask_genes: true` to hide them if you want to rule out keyword lookup.
