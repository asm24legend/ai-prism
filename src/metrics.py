"""Masked metrics: each gene is scored only on patients whose status is known (Issue 3)."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .data import GENES


def gene_auroc(y, p):
    keep = ~np.isnan(y)
    y, p = y[keep], p[keep]
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def evaluate(Y, P, threshold=0.5):
    res = {}
    for j, g in enumerate(GENES):
        keep = ~np.isnan(Y[:, j])
        y, p = Y[keep, j].astype(int), P[keep, j]
        both = len(np.unique(y)) == 2
        res[g] = {"auroc": float(roc_auc_score(y, p)) if both else float("nan"),
                  "auprc": float(average_precision_score(y, p)) if both else float("nan"),
                  "f1_at_0.5": float(f1_score(y, (p >= threshold).astype(int), zero_division=0)) if len(y) else float("nan"),
                  "n_known": int(keep.sum()), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    for k in ("auroc", "auprc", "f1_at_0.5"):
        res[f"macro_{k}"] = float(np.nanmean([res[g][k] for g in GENES]))
    return res


def bootstrap_auroc_ci(y, p, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for AUROC over patients (stratified resampling keeps both classes present)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def pooled_oof_report(Y, P, n_boot=1000, seed=0):
    """Pooled out-of-fold AUROC/AUPRC per gene with bootstrap CI (the headline numbers)."""
    out = {}
    for j, g in enumerate(GENES):
        keep = ~np.isnan(Y[:, j]) & ~np.isnan(P[:, j])
        y, p = Y[keep, j].astype(int), P[keep, j]
        if len(np.unique(y)) < 2:
            out[g] = {"auroc": float("nan"), "n_known": int(keep.sum())}
            continue
        lo, hi = bootstrap_auroc_ci(y, p, n_boot, seed + j)
        out[g] = {"auroc": float(roc_auc_score(y, p)), "auroc_ci95": [lo, hi],
                  "auprc": float(average_precision_score(y, p)), "n_known": int(keep.sum()), "n_pos": int(y.sum())}
    out["macro_auroc"] = float(np.nanmean([out[g]["auroc"] for g in GENES]))
    return out
