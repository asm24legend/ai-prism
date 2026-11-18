import numpy as np
import pandas as pd
import pytest
import torch

from src.chunking import window_ids
from src.data import GENES, parse_mut_genes, make_splits, site_of
from src.encoder_check import trained_position_length
from src.losses import masked_bce, no_information_floor
from src.metrics import evaluate
from src.model import ChunkPooler
from src.preprocess import mask_gene_text


def test_windows_cover_everything_and_overlap():
    ids = list(range(300))
    w = window_ids(ids, 125, 16)
    assert w[0][0] == 0 and w[-1][-1] == 299 and all(len(c) <= 125 for c in w)
    assert set(x for c in w for x in c) == set(ids)
    assert w[1][0] == 125 - 16
    assert window_ids([], 10, 2) == [[]]


def test_unknown_is_not_zero_and_masked_loss_ignores_it():
    logits = torch.zeros(2, 5, requires_grad=True)
    y = torch.tensor([[1., 0., float("nan"), 1., 0.], [float("nan")] * 5])
    masked_bce(logits, y).backward()
    g = logits.grad
    assert g[0, 2] == 0 and (g[1] == 0).all() and g[0, 0] != 0


def test_masked_metrics_only_score_known():
    Y = np.array([[1, np.nan, 0, 1, 0]] * 3 + [[0, 1, 1, 0, 1]] * 3, dtype=float)
    Y[0, 0] = np.nan
    P = np.random.rand(6, 5)
    r = evaluate(Y, P)
    assert r["KRAS"]["n_known"] == 5 and r["MUC16"]["n_known"] == 3


def test_parse_mut_genes():
    assert parse_mut_genes("KRAS; TP53")[0] == {"KRAS", "TP53"}
    assert parse_mut_genes(float("nan"))[0] == set()
    with pytest.raises(ValueError):
        parse_mut_genes("KRAS G12C; TP53")


def test_trained_position_length_detects_cutoff():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(768, generator=g)
    trained = base + 0.7 * torch.randn(127, 768, generator=g)       # neighbouring slots share structure
    untrained = torch.randn(1000, 768, generator=g)                  # random initial values
    assert trained_position_length(torch.cat([trained, untrained])) == 127
    assert trained_position_length(trained) == 127                   # no drop -> all slots trusted


def test_pooler_matches_manual():
    v = torch.randn(5, 4); owner = torch.tensor([0, 0, 1, 1, 1])
    assert torch.allclose(ChunkPooler("mean", 4)(v, owner, 2)[0], v[:2].mean(0), atol=1e-6)
    assert torch.allclose(ChunkPooler("max", 4)(v, owner, 2)[1], v[2:].max(0).values, atol=1e-6)
    a = ChunkPooler("attn", 4)(v, owner, 2)
    assert a.shape == (2, 4) and torch.isfinite(a).all()


def test_mask_genes():
    t = mask_gene_text("KRAS mutation G12C detected; TP53 wild type, T2N0")
    assert "KRAS" not in t and "G12C" not in t and "TP53" not in t and "T2N0" in t


def test_site_grouped_splits_never_share_a_site():
    rng = np.random.default_rng(0)
    Y = (rng.random((120, 5)) < 0.5).astype(float)
    sites = rng.choice(list("abcdefgh"), 120)
    for tr, te in make_splits(Y, sites, 4, 0, "site_grouped"):
        assert not set(sites[tr]) & set(sites[te])
