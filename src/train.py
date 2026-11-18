"""Fine-tune Clinical ModernBERT (windowed) + per-gene heads. Run only if a frozen baseline shows signal (Issue 9).

    python -m src.train --config configs/finetune.yaml
Fixed schedule (no per-fold epoch selection, no thresholds tuned on tiny folds): headline metrics are pooled
out-of-fold AUROC/AUPRC per gene with bootstrap CIs, scored only on patients with known labels.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .chunking import Chunker
from .config import load_config
from .data import GENES, make_splits
from .encoder_check import describe, inspect_encoder, resolve_window
from .losses import make_pos_weight, masked_bce, no_information_floor
from .metrics import evaluate, gene_auroc, pooled_oof_report
from .model import PRISMModel, set_trainable
from .pipeline import load_dataset, record_run


class PatientDataset(Dataset):
    def __init__(self, chunks, Y):
        self.chunks, self.Y = chunks, np.asarray(Y, dtype=np.float32)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i], self.Y[i]


def make_collate(pad_id):
    def collate(batch):
        flat, owner = [], []
        for b, (chunks, _) in enumerate(batch):
            flat += chunks
            owner += [b] * len(chunks)
        L = max(len(c) for c in flat)
        ids = torch.tensor([c + [pad_id] * (L - len(c)) for c in flat])
        mask = torch.tensor([[1] * len(c) + [0] * (L - len(c)) for c in flat])
        return {"input_ids": ids, "attention_mask": mask, "owner": torch.tensor(owner),
                "n": len(batch), "labels": torch.tensor(np.stack([y for _, y in batch]))}
    return collate


def fwd(model, b, device):
    return model(b["input_ids"].to(device), b["attention_mask"].to(device), b["owner"].to(device), b["n"])


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    out = []
    for b in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(torch.sigmoid(fwd(model, b, device).float()).cpu().numpy())
    return np.concatenate(out)


def build_optimizer(model, cfg, n_epochs, steps_per_epoch):
    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": other, "lr": cfg.lr_head}]
    if enc:
        groups.append({"params": enc, "lr": cfg.lr_encoder})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    total = max(1, steps_per_epoch * n_epochs)
    return opt, get_linear_schedule_with_warmup(opt, int(cfg.warmup * total), total)


def train_epoch(model, loader, pos_weight, opt, sched, scaler, device, amp_dtype, cfg):
    """Returns mean loss and the training-set predictions made during the epoch (for training AUROC)."""
    model.train()
    n_b, run, probs, ys = len(loader), 0.0, [], []
    opt.zero_grad()
    for i, b in enumerate(loader):
        group_size = min(cfg.grad_accum, n_b - (i // cfg.grad_accum) * cfg.grad_accum)   # last group is smaller
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = fwd(model, b, device)
        y = b["labels"].to(device)
        loss = masked_bce(logits, y, pos_weight)
        ((scaler.scale(loss) if scaler else loss) / group_size).backward()
        run += loss.item()
        probs.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        ys.append(b["labels"].numpy())
        if (i + 1) % cfg.grad_accum == 0 or i + 1 == n_b:
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            (scaler.step(opt), scaler.update()) if scaler else opt.step()
            sched.step()
            opt.zero_grad()
    P, Y = np.concatenate(probs), np.concatenate(ys)
    return run / max(n_b, 1), float(np.nanmean([gene_auroc(Y[:, j], P[:, j]) for j in range(len(GENES))]))


def run_fold(cfg, chunks, Y, split, pad_id, fold_dir, device):
    tr, te = split
    fold_dir.mkdir(parents=True, exist_ok=True)
    collate = make_collate(pad_id)
    mk = lambda idx, shuf: DataLoader(PatientDataset([chunks[i] for i in idx], Y[idx]), batch_size=cfg.batch_size,
                                      shuffle=shuf, collate_fn=collate, num_workers=0)
    tr_loader, te_loader = mk(tr, True), mk(te, False)
    steps_per_epoch = math.ceil(len(tr_loader) / cfg.grad_accum)

    model = PRISMModel(cfg.model_name, GENES, cfg.head_type, cfg.head_hidden, cfg.dropout,
                       cfg.chunk_pooling, cfg.grad_ckpt).to(device)
    model.init_biases(np.nanmean(Y[tr], axis=0))
    pw = make_pos_weight(Y[tr], device) if cfg.pos_weight else None
    floor = no_information_floor(Y[tr], pw)
    print(f"  no-information loss floor (predict prevalence for everyone): {floor:.4f}")

    amp_dtype, scaler = None, None
    if device.type == "cuda":
        if torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype, scaler = torch.float16, torch.amp.GradScaler("cuda")

    phases = []
    if cfg.phase1_epochs > 0:
        phases.append(("phase1_heads_only", cfg.phase1_epochs, 0))
    if cfg.phase2_epochs > 0:
        phases.append(("phase2_finetune", cfg.phase2_epochs, cfg.unfreeze_layers))
    hist, ep = [], 0
    for name, n_ep, n_unfreeze in phases:
        set_trainable(model.encoder, n_unfreeze)
        opt, sched = build_optimizer(model, cfg, n_ep, steps_per_epoch)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  -- {name}: {n_ep} epochs, top {n_unfreeze} encoder layers trainable, {n_train / 1e6:.2f}M trainable params")
        for _ in range(n_ep):
            ep += 1
            loss, tr_auc = train_epoch(model, tr_loader, pw, opt, sched, scaler, device, amp_dtype, cfg)
            hist.append({"epoch": ep, "phase": name, "train_loss": loss, "train_auroc": tr_auc, "loss_floor": floor})
            print(f"  epoch {ep:02d} | loss {loss:.4f} (floor {floor:.4f}) | train AUROC {tr_auc:.4f}")

    P_te = predict(model, te_loader, device, amp_dtype)          # final epoch: nothing was selected on test data
    res = {"test": evaluate(Y[te], P_te), "history": hist}
    json.dump(res, open(fold_dir / "metrics.json", "w"), indent=2)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, P_te, te


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", help="key=value pairs that override the config")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    record_run(cfg, out, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_dataset(cfg, out)
    texts, Y = df["clean"].tolist(), df[GENES].values.astype(np.float32)

    info = inspect_encoder(cfg.model_name)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    window = resolve_window(cfg.window_tokens, info, tok)
    describe(info, window)
    chunks = Chunker(tok, window, cfg.window_overlap, cfg.max_chunks).encode_all(texts)
    cfg.unfreeze_layers = min(cfg.unfreeze_layers, info["layers"])

    splits = make_splits(Y, df["site"].values, cfg.n_folds, cfg.seed, cfg.cv_scheme, cfg.n_repeats)
    oof = np.full((cfg.n_repeats, *Y.shape), np.nan, dtype=np.float32)
    fold_res = []
    for k, split in enumerate(splits):
        rep = k // cfg.n_folds
        print(f"\n=== repeat {rep + 1}, fold {k % cfg.n_folds + 1}/{cfg.n_folds} | train {len(split[0])} test {len(split[1])} ===")
        res, P_te, te = run_fold(cfg, chunks, Y, split, tok.pad_token_id, out / f"fold{k}", device)
        oof[rep][te] = P_te
        fold_res.append(res["test"])

    P = np.nanmean(oof, axis=0)                                   # average over repeats
    summary = {"pooled_oof": pooled_oof_report(Y, P, cfg.n_boot, cfg.seed),
               "mean_over_folds": {m: float(np.nanmean([r[m] for r in fold_res])) for m in ("macro_auroc", "macro_auprc")},
               "label_counts": pd.read_csv(out / "label_counts.csv").to_dict("records")}
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    pd.DataFrame(P, columns=[f"p_{g}" for g in GENES]).assign(Patient_ID=df["Patient_ID"].values) \
        .to_csv(out / "oof_predictions.csv", index=False)
    print("\n=== pooled out-of-fold (known labels only) ===")
    print(json.dumps(summary["pooled_oof"], indent=2))


if __name__ == "__main__":
    main()
