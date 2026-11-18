"""Three-state labels, text join, site codes, and patient-level splits.

Label states per patient and gene:  1 = mutated,  0 = sequenced and not mutated,  NaN = unknown.
Unknown is NEVER turned into 0 (specification, Model B; review Issue 3).
"""
import re

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

GENES = ["KRAS", "MUC16", "RYR2", "TP53", "TTN"]
NONSILENT = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site",
             "In_Frame_Del", "In_Frame_Ins", "Translation_Start_Site", "Nonstop_Mutation"}
KRAS_HOTSPOT_RE = re.compile(r"^p\.[A-Z](12|13|61)[A-Z*]")


def site_of(patient_id):
    """'TCGA-05-5423' -> '05' (tissue source site)."""
    return str(patient_id).split("-")[1]


# ----------------------------- labels ----------------------------- #
def parse_mut_genes(cell):
    """'KRAS; TP53' -> ({'KRAS','TP53'}, [unrecognised tokens]). Raises on 'KRAS G12C'-style entries that
    would otherwise silently become 0 (Issue 10)."""
    if not isinstance(cell, str) or not cell.strip():
        return set(), []
    genes, other = set(), []
    for tok in (t.strip().upper() for t in cell.split(";")):
        if not tok:
            continue
        if tok in GENES:
            genes.add(tok)
        elif tok.split()[0] in GENES or re.split(r"[,/|]", tok)[0] in GENES:
            raise ValueError(f"Unparseable Mut_Genes token {tok!r}: looks like a panel gene with extra text "
                             f"or a different separator; fix the table or the parser.")
        else:
            other.append(tok)
    return genes, other


def _read_ids(path):
    t = pd.read_csv(path)
    col = "Patient_ID" if "Patient_ID" in t.columns else t.columns[0]
    return set(t[col].astype(str).str.strip())


def _qualifies(maf, gene, definition):
    m = maf[(maf["Hugo_Symbol"] == gene) & maf["Variant_Classification"].isin(NONSILENT)]
    if definition == "kras_hotspot" and gene == "KRAS":
        m = m[m["HGVSp_Short"].astype(str).str.match(KRAS_HOTSPOT_RE)]
    return set(m["patient"])


def load_labels(cfg):
    """Return the patient table with one float column per gene (1 / 0 / NaN), 'site', and 'TMB' (if available)."""
    df = pd.read_csv(cfg.csv_path)
    df["site"] = df["Patient_ID"].map(site_of)
    pid = df["Patient_ID"].astype(str).str.strip()

    if cfg.maf_path:                                       # ---- label provenance: mutation calls ----
        maf = pd.read_csv(cfg.maf_path, sep=None, engine="python", comment="#")
        maf["patient"] = maf["Tumor_Sample_Barcode"].astype(str).str[:12]
        seq = _read_ids(cfg.sequenced_ids_path) if cfg.sequenced_ids_path else set(maf["patient"])
        if not cfg.sequenced_ids_path:
            print("[warn] no sequenced_ids_path: patients present in the MAF are treated as sequenced; "
                  "sequenced patients with zero calls would be mis-labelled unknown.")
        for g in GENES:
            pos = _qualifies(maf, g, cfg.mutation_definition)
            df[g] = [1.0 if p in pos else (0.0 if p in seq else np.nan) for p in pid]
        ns = maf[maf["Variant_Classification"].isin(NONSILENT)].groupby("patient").size()
        df["TMB"] = [float(ns.get(p, 0)) if p in seq else np.nan for p in pid]
        print(f"[labels] source=MAF, definition={cfg.mutation_definition}, sequenced patients={len(seq)}")
    else:                                                  # ---- table mode: only mutated genes are listed ----
        parsed = [parse_mut_genes(c) for c in df["Mut_Genes"]]
        other = sorted({t for _, o in parsed for t in o})
        print(f"[labels] source=table. distinct non-panel tokens in Mut_Genes: {other or 'none'}")
        has_rec = np.array([len(g) > 0 for g, _ in parsed])
        tested_col = df.get("Tested_Genes")
        for g in GENES:
            col = []
            for i, (mut, _) in enumerate(parsed):
                if g in mut:
                    col.append(1.0)
                elif tested_col is not None:
                    tg = {x.strip().upper() for x in str(tested_col.iloc[i]).split(";")}
                    col.append(0.0 if g in tg else np.nan)
                elif cfg.assume_tested_if_recorded and has_rec[i]:
                    col.append(0.0)
                else:
                    col.append(np.nan)
            df[g] = col
        if tested_col is None:
            if cfg.acknowledge_label_assumption:
                print("[labels] policy (acknowledged): panel genes only; a listed patient's unlisted gene = 0 (whole-exome); "
                      "empty record = unknown. Non-panel genes are ignored.")
            else:
                print("[warn] table has no 'Tested_Genes' column and no MAF: a 0 means 'listed patient, gene not listed' "
                      "(assumes whole-exome data). Patients with an empty record are UNKNOWN, not 0. "
                      "Set acknowledge_label_assumption: true once the team has confirmed this.")
            if has_rec.all():
                print("[note] every patient has >=1 panel-gene mutation (cohort conditioned on the outcome): prevalences are "
                      "inflated and genes are negatively correlated; results do not transfer to unselected LUAD patients.")
        df["TMB"] = np.nan
    if cfg.tmb_path:
        t = pd.read_csv(cfg.tmb_path)
        df["TMB"] = df["Patient_ID"].map(dict(zip(t["Patient_ID"].astype(str), t["TMB"])))
    return df


def label_summary(df):
    """Known positives / negatives / unknown per gene (open problem 5)."""
    rows = []
    for g in GENES:
        y = df[g].values
        rows.append({"gene": g, "n_pos": int(np.nansum(y == 1)), "n_neg": int(np.nansum(y == 0)),
                     "n_unknown": int(np.isnan(y).sum())})
    return pd.DataFrame(rows)


# ----------------------------- text ----------------------------- #
def attach_text_from_table(df, path, text_col="text", key_col="patient_filename"):
    """Join report text onto the label table by file name (case-insensitive, '.pdf' ignored), falling back to
    patient ID. Patients with no text are dropped and listed."""
    path = str(path)
    t = pd.read_excel(path) if path.lower().endswith((".xls", ".xlsx")) else pd.read_csv(path)
    norm = lambda s: s.astype(str).str.strip().str.lower().str.replace(r"\.pdf$", "", regex=True)
    t["_key"] = norm(t[key_col])
    t["_pid"] = t["_key"].str.split(".").str[0]
    by_key, by_pid = dict(zip(t["_key"], t[text_col])), dict(zip(t["_pid"], t[text_col]))
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


# ----------------------------- splits ----------------------------- #
def _strat_matrix(Y):
    """Stratification matrix for 3-state labels: known-positive columns plus unknown indicators (non-constant only)."""
    S = np.hstack([(Y == 1), np.isnan(Y)]).astype(int)
    return S[:, S.std(0) > 0] if (S.std(0) > 0).any() else np.zeros((len(Y), 1), dtype=int)


def make_splits(Y, groups, n_folds, seed, scheme="multilabel", n_repeats=1):
    """List of (train_idx, test_idx) for patient-level CV, repeated n_repeats times with new shuffles.
    scheme='site_grouped': every site is entirely in train or entirely in test (Issue 6)."""
    n, out = len(Y), []
    for r in range(n_repeats):
        s = seed + 1000 * r
        if scheme == "site_grouped":
            strat = np.nansum(Y == 1, axis=1).astype(int)
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=s)
            it = cv.split(np.zeros(n), strat, groups=np.asarray(groups))
        else:
            it = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s).split(
                np.zeros(n), _strat_matrix(Y))
        out += [(tr, te) for tr, te in it]
    return out


def make_gene_splits(y, groups, n_folds, seed, scheme="multilabel", n_repeats=1):
    """Splits for ONE gene over patients with a known label (indices into the 'known' subset)."""
    out = []
    for r in range(n_repeats):
        s = seed + 1000 * r
        if scheme == "site_grouped":
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=s)
            it = cv.split(np.zeros(len(y)), y, groups=np.asarray(groups))
        else:
            it = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s).split(np.zeros(len(y)), y)
        out += [(tr, te) for tr, te in it]
    return out
