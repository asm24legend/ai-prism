"""
Train Clinical ModernBERT + per-gene MLP heads on LUAD (KRAS, MUC16, RYR2, TP53, TTN).

Run from the project root:
    python -m src.train --config config.yaml
    python -m src.train --config config.yaml --override n_folds=1 max_len=2048
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from .data import GENES, ReportDataset, attach_text_from_table, load_labels, make_collate, make_splits
from .losses import build_loss
from .model import PRISMModel, set_trainable


# ----------------------------- metrics ----------------------------- #
def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def evaluate(Y, P, thresholds=None):
    res = {}
    for j, g in enumerate(GENES):
        y, p = Y[:, j], P[:, j]
        thr = 0.5 if thresholds is None else thresholds[j]
        both = len(np.unique(y)) == 2
        res[g] = {
            "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
            "auprc": float(average_precision_score(y, p)) if both else float("nan"),
            "f1": float(f1_score(y, (p >= thr).astype(int), zero_division=0)),
            "threshold": float(thr),
            "n_pos": int(y.sum()),
        }
    for k in ("auroc", "auprc", "f1"):
        res[f"macro_{k}"] = float(np.nanmean([res[g][k] for g in GENES]))
    return res


# ----------------------------- training ----------------------------- #
@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    out = []
    for b in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def build_optimizer(model, cfg, n_epochs, steps_per_epoch):
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": list(model.heads.parameters()), "lr": cfg.lr_head}]
    if enc:
        groups.append({"params": enc, "lr": cfg.lr_encoder})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    total = max(1, steps_per_epoch * n_epochs)
    return opt, get_linear_schedule_with_warmup(opt, int(cfg.warmup * total), total)


def train_epoch(model, loader, criterion, opt, sched, scaler, device, amp_dtype, cfg):
    model.train()
    run, n = 0.0, 0
    opt.zero_grad()
    for i, b in enumerate(loader, 1):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        loss = criterion(logits, b["labels"].to(device)) / cfg.grad_accum
        (scaler.scale(loss) if scaler else loss).backward()
        run += loss.item() * cfg.grad_accum
        n += 1
        if i % cfg.grad_accum == 0 or i == len(loader):
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad()
    return run / max(n, 1)


def run_fold(cfg, texts, Y, split, tokenizer, fold_dir, device):
    tr, va, te = split
    fold_dir.mkdir(parents=True, exist_ok=True)
    collate = make_collate(tokenizer, cfg.max_len)
    mk = lambda idx, shuf: DataLoader(
        ReportDataset(texts[idx], Y[idx]), batch_size=cfg.batch_size if shuf else cfg.eval_batch_size,
        shuffle=shuf, collate_fn=collate, num_workers=0)
    tr_loader, va_loader, te_loader = mk(tr, True), mk(va, False), mk(te, False)
    steps_per_epoch = math.ceil(len(tr_loader) / cfg.grad_accum)

    model = PRISMModel(cfg.model_name, GENES, cfg.head_hidden, cfg.dropout,
                       cfg.pooling, cfg.grad_ckpt).to(device)
    model.init_biases(Y[tr].mean(0))
    criterion = build_loss(cfg, Y[tr].mean(0), device)

    amp_dtype, scaler = None, None
    if device.type == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:   # Ampere or newer; RTX 20-series lacks native bf16
            amp_dtype = torch.bfloat16
        else:
            amp_dtype, scaler = torch.float16, torch.amp.GradScaler("cuda")

    phases = []
    if cfg.phase1_epochs > 0:
        phases.append(("phase1_heads_only", cfg.phase1_epochs, 0))
    if cfg.phase2_epochs > 0:
        phases.append(("phase2_finetune", cfg.phase2_epochs, cfg.unfreeze_layers))

    best, hist, ep_global = -1.0, [], 0
    ckpt = fold_dir / "best.pt"
    for name, n_ep, n_unfreeze in phases:
        set_trainable(model.encoder, n_unfreeze)
        opt, sched = build_optimizer(model, cfg, n_ep, steps_per_epoch)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  -- {name}: {n_ep} epochs, top {n_unfreeze} encoder layers trainable, "
              f"{n_train / 1e6:.1f}M trainable params")
        bad = 0
        for _ in range(n_ep):
            ep_global += 1
            loss = train_epoch(model, tr_loader, criterion, opt, sched, scaler, device, amp_dtype, cfg)
            m = evaluate(Y[va], predict(model, va_loader, device, amp_dtype))
            hist.append({"epoch": ep_global, "phase": name, "train_loss": loss,
                         "val_macro_auprc": m["macro_auprc"], "val_macro_auroc": m["macro_auroc"]})
            print(f"  epoch {ep_global:02d} | loss {loss:.4f} | val AUPRC {m['macro_auprc']:.4f} "
                  f"| val AUROC {m['macro_auroc']:.4f}")
            if m["macro_auprc"] > best:
                best, bad = m["macro_auprc"], 0
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f"  early stop in {name}")
                    break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    P_va = predict(model, va_loader, device, amp_dtype)
    thr = [best_threshold(Y[va][:, j], P_va[:, j]) for j in range(len(GENES))]   # tuned on validation only
    P_te = predict(model, te_loader, device, amp_dtype)
    res = {"test_tuned_thr": evaluate(Y[te], P_te, thr), "test_thr_0.5": evaluate(Y[te], P_te),
           "val_best_macro_auprc": best, "history": hist}
    json.dump(res, open(fold_dir / "metrics.json", "w"), indent=2)
    if not cfg.keep_ckpt:
        ckpt.unlink(missing_ok=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, P_te, te


# ----------------------------- main ----------------------------- #
def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for o in overrides or []:
        k, v = o.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return argparse.Namespace(**cfg)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--override", nargs="*", help="key=value pairs that override the config")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    set_seed(cfg.seed)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(cfg), open(out / "config_used.json", "w"), indent=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_labels(cfg.csv_path)
    df = attach_text_from_table(df, cfg.reports_file, cfg.text_col, cfg.key_col)
    df["clean"] = df["Text_Report"].astype(str)
    short = df["clean"].str.len() < cfg.min_chars
    if short.any():
        print(f"[warn] dropping {int(short.sum())} reports with <{cfg.min_chars} chars: "
              f"{df.loc[short, 'Patient_ID'].tolist()}")
        df = df[~short].reset_index(drop=True)

    texts, Y = df["clean"].values, df[GENES].values.astype(np.float32)
    print(f"[info] {len(df)} patients | prevalence: " +
          ", ".join(f"{g}={Y[:, j].mean():.2f}" for j, g in enumerate(GENES)))

    mc = AutoConfig.from_pretrained(cfg.model_name)
    print(f"[info] encoder: model_type={mc.model_type}, layers={mc.num_hidden_layers}, "
          f"hidden={mc.hidden_size}, max_position_embeddings={mc.max_position_embeddings}")
    if cfg.max_len > mc.max_position_embeddings:
        print(f"[warn] max_len {cfg.max_len} exceeds the model's {mc.max_position_embeddings} positions; "
              f"capping. Longer reports will be truncated.")
        cfg.max_len = mc.max_position_embeddings
    cfg.unfreeze_layers = min(cfg.unfreeze_layers, mc.num_hidden_layers)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    splits = make_splits(Y, cfg.n_folds, cfg.seed)

    all_res, oof = [], np.full(Y.shape, np.nan, dtype=np.float32)
    for k, split in enumerate(splits):
        print(f"\n=== fold {k + 1}/{len(splits)} | train {len(split[0])} val {len(split[1])} test {len(split[2])} ===")
        res, P_te, te = run_fold(cfg, texts, Y, split, tokenizer, out / f"fold{k}", device)
        oof[te] = P_te
        all_res.append(res)

    summary = {}
    for key in ("test_tuned_thr", "test_thr_0.5"):
        summary[key] = {m: {"mean": float(np.nanmean([r[key][m] for r in all_res])),
                            "std": float(np.nanstd([r[key][m] for r in all_res]))}
                        for m in ("macro_auroc", "macro_auprc", "macro_f1")}
        summary[key]["per_gene_auroc"] = {g: float(np.nanmean([r[key][g]["auroc"] for r in all_res])) for g in GENES}
        summary[key]["per_gene_auprc"] = {g: float(np.nanmean([r[key][g]["auprc"] for r in all_res])) for g in GENES}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    pd.DataFrame(oof, columns=[f"p_{g}" for g in GENES]).assign(Patient_ID=df["Patient_ID"].values) \
        .to_csv(out / "oof_predictions.csv", index=False)
    print("\n=== summary (mean over folds) ===")
    print(json.dumps(summary["test_tuned_thr"], indent=2))


if __name__ == "__main__":
    main()