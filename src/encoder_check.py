"""Encoder sanity checks (Issue 2). Run before trusting any checkpoint.

Published Simonlee711/Clinical_ModernBERT loads as a classic BertModel whose 8,192-row position table was only
trained for slots 0-126 (CLS + 126 tokens). Text must therefore be fed in windows that stay inside the trained slots.
"""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM


def load_encoder(model_name, device_map=None):
    """Load the checkpoint the way its model card does (AutoModelForMaskedLM, optional device_map='auto') and return
    the transformer body without the MLM head: its hidden states are the features. Needs `accelerate` for device_map."""
    mlm = AutoModelForMaskedLM.from_pretrained(model_name, device_map=device_map) if device_map \
        else AutoModelForMaskedLM.from_pretrained(model_name)
    return mlm.base_model


def position_table(model):
    emb = getattr(model, "embeddings", None)
    pe = getattr(emb, "position_embeddings", None)
    return None if pe is None else pe.weight.detach().float()


def trained_position_length(pe, sim_threshold=0.2, run=8):
    """Number of leading position slots that were actually trained (neighbouring slots similar), read from the
    weights. Untrained slots keep random initial values: neighbouring similarity ~0. Returns len(pe) if no drop."""
    sim = F.cosine_similarity(pe[:-1], pe[1:], dim=1)
    for i in range(1, len(sim) - run):
        if bool((sim[i:i + run] < sim_threshold).all()):
            return i + 1                       # first slot pair that is dissimilar is (i, i+1): slots 0..i trained
    return len(pe)


def inspect_encoder(model_name):
    """Load the encoder once and report what it really is."""
    cfg = AutoConfig.from_pretrained(model_name)
    model = load_encoder(model_name)
    pe = position_table(model)
    info = {"model_type": cfg.model_type, "class": type(model).__name__, "hidden": cfg.hidden_size,
            "layers": cfg.num_hidden_layers, "max_position_embeddings": cfg.max_position_embeddings,
            "learned_position_table": pe is not None,
            "trained_positions": trained_position_length(pe) if pe is not None else None}
    del model
    return info


def resolve_window(requested, info, tokenizer, hard_cap_rotary=1024):
    """Text tokens per window. 'auto' = trusted positions - special tokens. An explicit number is checked against
    the trained positions and refused if it exceeds them."""
    from .chunking import special_affixes
    n_special = sum(len(a) for a in special_affixes(tokenizer))
    trusted = info["trained_positions"] or min(info["max_position_embeddings"], hard_cap_rotary)
    max_text = trusted - n_special
    if requested in (None, "auto"):
        return max_text
    if int(requested) > max_text:
        raise ValueError(f"window_tokens={requested} exceeds the {max_text} text tokens this checkpoint was trained "
                         f"for ({trusted} positions incl. {n_special} special). Use 'auto' or a smaller number.")
    return int(requested)


def describe(info, window):
    print(f"[encoder] class={info['class']} model_type={info['model_type']} layers={info['layers']} "
          f"hidden={info['hidden']} declared_positions={info['max_position_embeddings']}")
    if info["learned_position_table"]:
        print(f"[encoder] learned position table: only slots 0..{info['trained_positions'] - 1} are trained "
              f"-> window = {window} text tokens (+ special tokens)")
        if info["class"] != "ModernBertModel":
            print("[warn] checkpoint loads as a classic BERT, not ModernBERT: no long-context ability. "
                  "Text is chunked; do not raise window_tokens.")
    else:
        print(f"[encoder] no learned position table (rotary): window = {window} text tokens. "
              f"Verify with tools/check_encoder.py before trusting long windows.")


@torch.no_grad()
def masked_word_accuracy(model_name, passage, filler, max_words=None):
    """Appendix A1: fill-in-the-blank accuracy for the same passage at the start vs after `filler` tokens.
    Accuracy must not collapse when the passage sits later in the input."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mlm = AutoModelForMaskedLM.from_pretrained(model_name).eval()
    p = tok(passage, add_special_tokens=False)["input_ids"]
    f = tok(filler, add_special_tokens=False)["input_ids"]

    def acc(prefix, suffix):
        hits = n = 0
        for i, gold in enumerate(p):
            if not tok.decode([gold]).strip().isalpha():
                continue
            ids = [tok.cls_token_id] + prefix + p + suffix + [tok.sep_token_id]
            j = 1 + len(prefix) + i
            ids[j] = tok.mask_token_id
            pred = mlm(torch.tensor([ids])).logits[0, j].argmax().item()
            hits += pred == gold
            n += 1
            if max_words and n >= max_words:
                break
        return hits / max(n, 1)

    return {"passage_first": acc([], []), "passage_after_filler": acc(f, []), "passage_first_filler_after": acc([], f)}
