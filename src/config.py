"""Config loading with explicit defaults. One YAML per experiment (see configs/), never edited after a run."""
import argparse
import yaml

DEFAULTS = dict(
    # ---- data ----
    csv_path="data/PRISM_LUAD_Dataset_-_Sheet1.csv",
    reports_file="data/TCGA_Reports.xlsx",
    text_col="text", key_col="patient_filename", min_chars=100,
    # three-state labels (Issue 3). Without maf_path the label table only lists mutated genes.
    maf_path=None,                 # MAF with Tumor_Sample_Barcode, Hugo_Symbol, Variant_Classification, HGVSp_Short
    sequenced_ids_path=None,       # one patient ID per line (or CSV with Patient_ID): patients with sequencing data
    mutation_definition="nonsilent",   # "nonsilent" | "kras_hotspot" (KRAS codons 12/13/61, others non-silent); Issue 1
    assume_tested_if_recorded=True,    # table mode only: patient with a non-empty record counts as sequenced (WES)
    acknowledge_label_assumption=False,  # True = the team has decided that 'not listed' means 'not mutated' (WES); shortens the warning
    tmb_path=None,                 # optional CSV Patient_ID,TMB (else TMB is computed from the MAF if given)
    # ---- text preprocessing (Issues 6, 7, 10) ----
    clean_text=True, mask_genes=False, mask_variants=True, mask_p53_ihc=False,
    strip_site_boilerplate=False, boilerplate_min_frac=0.5, boilerplate_min_reports=4,
    # ---- run ----
    output_dir="runs/exp", seed=42, n_folds=5, n_repeats=1,
    cv_scheme="multilabel",        # "multilabel" (iterative stratification) | "site_grouped" (StratifiedGroupKFold)
    # ---- encoder / chunking (Issue 2) ----
    model_name="Simonlee711/Clinical_ModernBERT",
    window_tokens="auto",          # text tokens per window; "auto" = trusted positions minus special tokens
    window_overlap=10, max_chunks=24,
    chunk_pooling="mean",          # mean | max | attn  (pooling of chunk vectors into the report vector)
    embed_batch=32, embeddings_path=None,
    device_map="auto",             # embed only (needs `accelerate`); null = plain .to(device). Fine-tuning always uses .to(device)
    # ---- heads / loss ----
    head_type="linear",            # linear (specification) | mlp
    head_hidden=128, dropout=0.1, pos_weight=False, grad_ckpt=True,
    # ---- fine-tuning (Issue 8): fixed schedule, no per-fold epoch selection, no thresholds tuned on tiny folds ----
    batch_size=8, grad_accum=1, weight_decay=0.01, warmup=0.1, max_grad_norm=1.0,
    lr_head=1e-3, lr_encoder=2e-5, phase1_epochs=30, phase2_epochs=0, unfreeze_layers=2,
    # ---- baselines (Issue 9) ----
    variants=["raw"], lr_C=0.05, tfidf_C=10.0, lowdim_C=1.0, n_boot=1000, n_perm=200, perm_models=["tfidf_lr", "emb_lr"],
    tfidf_max_features=20000, baseline_models=["prevalence", "site_lr", "tmb_lr", "tfidf_lr", "emb_lr", "emb_tmb_lr"],
)


def load_config(path, overrides=None):
    cfg = dict(DEFAULTS)
    user = yaml.safe_load(open(path)) or {}
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config keys in {path}: {sorted(unknown)}")
    cfg.update(user)
    for o in overrides or []:
        k, v = o.split("=", 1)
        if k not in DEFAULTS:
            raise KeyError(f"Unknown override key: {k}")
        cfg[k] = yaml.safe_load(v)
    return argparse.Namespace(**cfg)
