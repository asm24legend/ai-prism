"""Data loading, text join, Dataset/collate, and patient-level multilabel-stratified splits."""
import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import (MultilabelStratifiedKFold,
                                      MultilabelStratifiedShuffleSplit)
from torch.utils.data import Dataset

GENES = ["KRAS", "MUC16", "RYR2", "TP53", "TTN"]


def load_labels(csv_path):
    """LUAD table -> one 0/1 column per gene, parsed from 'KRAS; TP53; ...'."""
    df = pd.read_csv(csv_path)
    df["Mut_Genes"] = df["Mut_Genes"].fillna("")
    for g in GENES:
        df[g] = df["Mut_Genes"].apply(
            lambda s: int(g in [x.strip().upper() for x in str(s).split(";")]))
    return df


def attach_text_from_table(df, path, text_col="text", key_col="patient_filename"):
    """Join report text from TCGA_Reports.xlsx onto the LUAD table.
    Match on the full file name (case-insensitive, '.pdf' ignored); fall back to patient ID.
    Patients with no text are dropped and listed."""
    path = str(path)
    t = pd.read_excel(path) if path.lower().endswith((".xls", ".xlsx")) else pd.read_csv(path)
    norm = lambda s: s.astype(str).str.strip().str.lower().str.replace(r"\.pdf$", "", regex=True)
    t["_key"] = norm(t[key_col])
    t["_pid"] = t["_key"].str.split(".").str[0]
    by_key = dict(zip(t["_key"], t[text_col]))
    by_pid = dict(zip(t["_pid"], t[text_col]))      # assumes one report per patient
    keys, pids = norm(df["Text_Report"]), df["Patient_ID"].astype(str).str.strip().str.lower()
    texts = []
    for k, p in zip(keys, pids):
        x = by_key.get(k, by_pid.get(p))
        texts.append(x if isinstance(x, str) else None)
    df = df.copy()
    df["Text_Report"] = texts
    missing = df[df["Text_Report"].isna()]
    if len(missing):
        print(f"[warn] no report text for {len(missing)} patients, dropping: {missing['Patient_ID'].tolist()}")
    return df.dropna(subset=["Text_Report"]).reset_index(drop=True)


class ReportDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = list(texts), np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_len):
    def collate(batch):
        texts, ys = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(np.stack(ys))
        return enc
    return collate


def _carve_val(trv_idx, Y, frac, seed):
    s = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    tr_rel, va_rel = next(s.split(np.zeros(len(trv_idx)), Y[trv_idx]))
    return trv_idx[tr_rel], trv_idx[va_rel]


def make_splits(Y, n_folds, seed, val_frac=0.15):
    """Multilabel-stratified (train_idx, val_idx, test_idx) triples. One row = one patient."""
    n = len(Y)
    if n_folds <= 1:
        s = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        trv, te = next(s.split(np.zeros(n), Y))
        tr, va = _carve_val(trv, Y, 0.15 / 0.85, seed)
        return [(tr, va, te)]
    kf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = []
    for trv, te in kf.split(np.zeros(n), Y):
        tr, va = _carve_val(trv, Y, val_frac, seed)
        out.append((tr, va, te))
    return out
