"""Text cleaning, gene-name masking (Issue 7) and optional site-boilerplate stripping (Issue 6)."""
import re
import unicodedata
from collections import Counter, defaultdict

GENE_RE = re.compile(r"\b(K-?RAS|TP53|TTN|titin|MUC-?16|CA-?125|RYR-?2)\b", re.I)
P53_RE = re.compile(r"\bp53\b", re.I)
VARIANT_RE = re.compile(r"\b[A-Z]\d{1,4}[A-Z*]\b")          # e.g. G12C, R273H
PAGE_RE = re.compile(r"^\s*page\s+\d+\s*(of|/)\s*\d+\s*$", re.I)


def clean_text(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.splitlines()]
    lines = [ln for ln in lines if ln and not PAGE_RE.match(ln)]
    return "\n".join(lines)


def mask_gene_text(t, mask_variants=True, mask_p53_ihc=False):
    t = GENE_RE.sub("[GENE]", t)
    if mask_p53_ihc:
        t = P53_RE.sub("[GENE]", t)
    if mask_variants:
        t = VARIANT_RE.sub("[VAR]", t)
    return t


def gene_mentions(texts, ids, width=40):
    """One row per report that names a panel gene: what it says (hand-check these, Issue 7)."""
    rows = []
    for pid, t in zip(ids, texts):
        hits = [(m.group(0), t[max(0, m.start() - width): m.end() + width].replace("\n", " "))
                for m in GENE_RE.finditer(t)]
        if hits:
            rows.append({"Patient_ID": pid, "n_hits": len(hits), "matches": "; ".join(sorted({h[0] for h in hits})),
                         "context": " || ".join(h[1] for h in hits[:3])})
    return rows


def strip_site_boilerplate(texts, sites, min_frac=0.5, min_reports=4):
    """Remove lines that occur verbatim in >= min_frac of the reports of one site (template text, headers,
    disclaimers). Only sites with >= min_reports reports are processed. Unsupervised: uses no labels."""
    by_site = defaultdict(list)
    for i, s in enumerate(sites):
        by_site[s].append(i)
    out = list(texts)
    for s, idx in by_site.items():
        if len(idx) < min_reports:
            continue
        cnt = Counter()
        for i in idx:
            cnt.update({ln.strip().lower() for ln in texts[i].splitlines()})
        common = {ln for ln, c in cnt.items() if c / len(idx) >= min_frac and ln}
        for i in idx:
            out[i] = "\n".join(ln for ln in texts[i].splitlines() if ln.strip().lower() not in common)
    return out


def preprocess_texts(texts, sites, cfg):
    texts = [str(t) for t in texts]
    if cfg.clean_text:
        texts = [clean_text(t) for t in texts]
    if cfg.strip_site_boilerplate:
        texts = strip_site_boilerplate(texts, sites, cfg.boilerplate_min_frac, cfg.boilerplate_min_reports)
    if cfg.mask_genes:
        texts = [mask_gene_text(t, cfg.mask_variants, cfg.mask_p53_ihc) for t in texts]
    return texts
