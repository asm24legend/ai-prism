"""Encode every report ONCE with the frozen encoder and cache the window vectors (Issues 2, 8, 9).

    python -m src.embed --config configs/frozen_baselines.yaml
Writes <output_dir>/embeddings_<raw|masked>.npz (or embeddings_path). Pool with src.model.pool_numpy.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .chunking import Chunker
from .config import load_config
from .encoder_check import describe, inspect_encoder, load_encoder, resolve_window
from .pipeline import load_dataset, record_run


def cache_path(cfg, variant):
    return Path(cfg.embeddings_path) if cfg.embeddings_path else Path(cfg.output_dir) / f"embeddings_{variant}.npz"


def load_embeddings(path, patient_ids):
    z = np.load(path, allow_pickle=True)
    ids, off, vecs = list(z["patient_ids"]), z["offsets"], z["vecs"]
    pos = {p: i for i, p in enumerate(ids)}
    return [vecs[off[pos[p]]:off[pos[p] + 1]] for p in patient_ids]


@torch.no_grad()
def embed_texts(texts, tokenizer, encoder, chunker, batch, device):
    chunks = chunker.encode_all(texts)
    flat = [(i, c) for i, cs in enumerate(chunks) for c in cs]
    vecs = np.zeros((len(flat), encoder.config.hidden_size), dtype=np.float32)
    pad = tokenizer.pad_token_id
    for s in range(0, len(flat), batch):
        part = [c for _, c in flat[s:s + batch]]
        L = max(len(c) for c in part)
        ids = torch.tensor([c + [pad] * (L - len(c)) for c in part], device=device)
        mask = torch.tensor([[1] * len(c) + [0] * (L - len(c)) for c in part], device=device)
        h = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        vecs[s:s + len(part)] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
    counts = np.bincount([i for i, _ in flat], minlength=len(texts))
    return vecs, np.concatenate([[0], np.cumsum(counts)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    out = Path(cfg.output_dir)
    record_run(cfg, out, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = inspect_encoder(cfg.model_name)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    window = resolve_window(cfg.window_tokens, info, tok)
    describe(info, window)
    encoder = load_encoder(cfg.model_name, cfg.device_map)      # AutoModelForMaskedLM(..., device_map) body, no MLM head
    if not cfg.device_map:
        encoder = encoder.to(device)
    encoder.eval()
    device = next(encoder.parameters()).device                  # inputs go to the encoder's first device
    for variant in cfg.variants:
        df = load_dataset(cfg, out, mask_override=(variant == "masked"))
        chunker = Chunker(tok, window, cfg.window_overlap, cfg.max_chunks)
        vecs, off = embed_texts(df["clean"].tolist(), tok, encoder, chunker, cfg.embed_batch, device)
        p = cache_path(cfg, variant)
        np.savez(p, patient_ids=np.array(df["Patient_ID"]), offsets=off, vecs=vecs, window=window,
                 overlap=cfg.window_overlap, model=cfg.model_name)
        print(f"[embed] {variant}: saved {vecs.shape} window vectors for {len(df)} patients -> {p}")


if __name__ == "__main__":
    main()
