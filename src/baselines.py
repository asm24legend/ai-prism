"""Cheap baselines under repeated (optionally site-grouped) CV, run BEFORE any fine-tuning (Issues 1, 6, 7, 9).

    python -m src.embed     --config configs/frozen_baselines.yaml     # once, for the embedding baselines
    python -m src.baselines --config configs/frozen_baselines.yaml

Models: prevalence | site_lr | tmb_lr | tfidf_lr | emb_lr | emb_tmb_lr. Per gene, only patients with a known label.
Reports pooled out-of-fold AUROC (averaged over repeats), bootstrap 95% CI, label-permutation p-value, and
text+TMB vs TMB-only AUROC difference. Bonferroni threshold = 0.05 / number of genes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_config
from .data import GENES, make_gene_splits
from .embed import cache_path, load_embeddings
from .metrics import bootstrap_auroc_ci
from .model import pool_numpy
from .pipeline import load_dataset, record_run


def _lr(C):
    return LogisticRegression(C=C, max_iter=5000)


def cv_oof(X, y, splits, C, scale):
    """Out-of-fold probabilities averaged over repeats. X dense or sparse; scaler fit inside each training fold."""
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = _lr(C).fit(Xtr, y[tr])
        acc[te] += m.predict_proba(Xte)[:, 1]
        cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def model_C(name, cfg):
    """L2 strength matters: too strong a penalty makes fold predictions near-constant, so pooled AUROC is then
    dominated by fold-to-fold intercept differences. Sparse tf-idf and 1-3 dim features need weak penalties."""
    return {"tfidf_lr": cfg.tfidf_C, "emb_lr": cfg.lr_C, "emb_tmb_lr": cfg.lr_C}.get(name, cfg.lowdim_C)


def build_features(name, df, cfg, variant, tfidf_X, emb_pooled):
    if name == "site_lr":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(df[["site"]]), False
    if name == "tmb_lr":
        return np.log1p(df[["TMB"]].values.astype(float)), True
    if name == "tfidf_lr":
        return tfidf_X, False
    if name == "emb_lr":
        return emb_pooled, True
    if name == "emb_tmb_lr":
        tmb = np.log1p(df[["TMB"]].values.astype(float))
        return np.hstack([emb_pooled, np.repeat(tmb, 8, axis=1)]), True      # TMB repeated so it survives L2 shrinkage
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    out = Path(cfg.output_dir)
    record_run(cfg, out, args.config)
    rng = np.random.default_rng(cfg.seed)
    rows, delta_rows = [], []

    for variant in cfg.variants:
        df = load_dataset(cfg, out, mask_override=(variant == "masked"))
        has_tmb = df["TMB"].notna().all()
        if not has_tmb:
            print("[warn] TMB unavailable for all patients (give maf_path or tmb_path): TMB baselines skipped.")
        models = [m for m in cfg.baseline_models if has_tmb or "tmb" not in m]
        tfidf_X = emb_pooled = None
        if "tfidf_lr" in models:    # IDF is fit on all texts without labels (unsupervised, no label leakage)
            tfidf_X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                      max_features=cfg.tfidf_max_features).fit_transform(df["clean"]).tocsr()
        if {"emb_lr", "emb_tmb_lr"} & set(models):
            p = cache_path(cfg, variant)
            if not p.exists():
                print(f"[warn] {p} not found (run src.embed first): embedding baselines skipped for '{variant}'.")
                models = [m for m in models if not m.startswith("emb")]
            else:
                emb_pooled = np.stack([pool_numpy(v, cfg.chunk_pooling) for v in load_embeddings(p, df["Patient_ID"])])

        for j, g in enumerate(GENES):
            known = ~df[g].isna().values
            y = df[g].values[known].astype(int)
            if min((y == 1).sum(), (y == 0).sum()) < 5:
                print(f"[skip] {g}: too few known positives/negatives")
                continue
            sub = df[known].reset_index(drop=True)
            splits = make_gene_splits(y, sub["site"].values, cfg.n_folds, cfg.seed, cfg.cv_scheme, cfg.n_repeats)
            oofs = {}
            for m in models:
                if m == "prevalence":
                    oofs[m] = np.full(len(y), y.mean()); auc = 0.5; lo = hi = 0.5; pval = np.nan
                else:
                    X, scale = build_features(m, sub, cfg, variant,
                                              None if tfidf_X is None else tfidf_X[known],
                                              None if emb_pooled is None else emb_pooled[known])
                    C = model_C(m, cfg)
                    oofs[m] = cv_oof(X, y, splits, C, scale)
                    auc = roc_auc_score(y, oofs[m])
                    lo, hi = bootstrap_auroc_ci(y, oofs[m], cfg.n_boot, cfg.seed + j)
                    pval = np.nan
                    if m in cfg.perm_models and cfg.n_perm > 0:
                        null = np.array([roc_auc_score(yp := rng.permutation(y), cv_oof(X, yp, splits, C, scale))
                                         for _ in range(cfg.n_perm)])
                        pval = float((1 + (null >= auc).sum()) / (1 + cfg.n_perm))
                rows.append({"variant": variant, "gene": g, "model": m, "n_known": len(y), "n_pos": int(y.sum()),
                             "auroc": auc, "ci_lo": lo, "ci_hi": hi, "perm_p": pval,
                             "bonferroni_sig": (pval < 0.05 / len(GENES)) if pval == pval else None})
                print(f"[{variant}] {g:5s} {m:11s} AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]"
                      + (f"  perm p={pval:.4f}" if pval == pval else ""))
            if "emb_tmb_lr" in oofs and "tmb_lr" in oofs:      # does text add anything beyond mutation burden?
                d = []
                pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
                for _ in range(cfg.n_boot):
                    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                    d.append(roc_auc_score(y[idx], oofs["emb_tmb_lr"][idx]) - roc_auc_score(y[idx], oofs["tmb_lr"][idx]))
                delta_rows.append({"variant": variant, "gene": g,
                                   "delta_auroc": roc_auc_score(y, oofs["emb_tmb_lr"]) - roc_auc_score(y, oofs["tmb_lr"]),
                                   "ci_lo": float(np.quantile(d, 0.025)), "ci_hi": float(np.quantile(d, 0.975))})

    res = pd.DataFrame(rows)
    res.to_csv(out / "baseline_results.csv", index=False)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(out / "text_plus_tmb_vs_tmb.csv", index=False)
    print("\n" + res.pivot_table(index=["variant", "model"], columns="gene", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
