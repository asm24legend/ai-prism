"""Shared data pipeline: labels -> text -> preprocessing -> filtered patient table. Used by embed, baselines, train."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .data import GENES, attach_text_from_table, label_summary, load_labels
from .preprocess import gene_mentions, preprocess_texts


def git_state():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
        return {"commit": h, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def record_run(cfg, out: Path, config_path):
    """Reproducibility record (Issue 5): exact config, its source file, git commit, library versions."""
    import shutil, torch, transformers, sklearn
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(cfg), open(out / "config_used.json", "w"), indent=2, default=str)
    shutil.copy(config_path, out / "config_source.yaml")
    json.dump({**git_state(), "torch": torch.__version__, "transformers": transformers.__version__,
               "sklearn": sklearn.__version__}, open(out / "run_info.json", "w"), indent=2)


def load_dataset(cfg, out: Path = None, mask_override=None):
    """Return df with columns: Patient_ID, site, GENES (1/0/NaN), TMB, raw, clean. mask_override forces the
    gene-masking flag (used by baselines to run raw vs masked)."""
    df = load_labels(cfg)
    df = attach_text_from_table(df, cfg.reports_file, cfg.text_col, cfg.key_col)
    df["raw"] = df["Text_Report"].astype(str)
    if out is not None:                                       # Issue 7: which reports name a gene
        rows = gene_mentions(df["raw"].tolist(), df["Patient_ID"].tolist())
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "gene_mentions.csv", index=False)
        print(f"[info] {len(rows)} of {len(df)} reports name a panel gene (see gene_mentions.csv)")
    c = cfg
    if mask_override is not None:
        import copy
        c = copy.copy(cfg)
        c.mask_genes = mask_override
    df["clean"] = preprocess_texts(df["raw"].tolist(), df["site"].tolist(), c)
    short = df["clean"].str.len() < cfg.min_chars
    if short.any():
        print(f"[warn] dropping {int(short.sum())} reports with <{cfg.min_chars} chars: {df.loc[short, 'Patient_ID'].tolist()}")
        df = df[~short]
    unknown_all = df[GENES].isna().all(axis=1)
    if unknown_all.any():
        print(f"[warn] {int(unknown_all.sum())} patients have NO known label for any gene and are excluded: "
              f"{df.loc[unknown_all, 'Patient_ID'].tolist()}")
        df = df[~unknown_all]
    df = df.reset_index(drop=True)
    ls = label_summary(df)
    print("[labels] known positives / negatives / unknown per gene:\n" + ls.to_string(index=False))
    for _, r in ls.iterrows():
        if min(r.n_pos, r.n_neg) < 10:
            print(f"[warn] {r.gene}: fewer than 10 known {'positives' if r.n_pos < r.n_neg else 'negatives'}")
    if out is not None:
        ls.to_csv(out / "label_counts.csv", index=False)
    return df
