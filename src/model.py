"""Clinical ModernBERT on report windows -> report vector -> one head per gene."""
import numpy as np
import torch
import torch.nn as nn
from .data import GENES
from .encoder_check import load_encoder


class ChunkPooler(nn.Module):
    """Pool the vectors of a patient's windows into one report vector. mean | max | attn."""
    def __init__(self, mode, d):
        super().__init__()
        self.mode = mode
        self.score = nn.Linear(d, 1) if mode == "attn" else None

    def forward(self, v, owner, n):                       # v (N, d), owner (N,) in [0, n)
        d = v.shape[1]
        idx = owner.unsqueeze(1).expand(-1, d)
        if self.mode == "max":
            out = torch.full((n, d), float("-inf"), device=v.device, dtype=v.dtype)
            return out.scatter_reduce(0, idx, v, reduce="amax", include_self=True)
        if self.mode == "attn":
            s = self.score(v).squeeze(-1)
            smax = torch.full((n,), float("-inf"), device=v.device, dtype=s.dtype).scatter_reduce(
                0, owner, s, reduce="amax", include_self=True)
            e = torch.exp(s - smax[owner])
            den = torch.zeros(n, device=v.device, dtype=e.dtype).index_add(0, owner, e)
            w = (e / den[owner]).unsqueeze(1)
            return torch.zeros(n, d, device=v.device, dtype=v.dtype).index_add(0, owner, v * w)
        cnt = torch.zeros(n, device=v.device, dtype=v.dtype).index_add(0, owner, torch.ones_like(owner, dtype=v.dtype))
        return torch.zeros(n, d, device=v.device, dtype=v.dtype).index_add(0, owner, v) / cnt.unsqueeze(1)


def pool_numpy(chunk_vecs, mode="mean"):
    """Same pooling for cached embeddings (attn is learned, so only mean/max here)."""
    return chunk_vecs.mean(0) if mode == "mean" else chunk_vecs.max(0)


class GeneHead(nn.Module):
    """'linear' = the specification's p_i = sigmoid(w_i^T z + b_i); 'mlp' = small gene-specific network."""
    def __init__(self, d, kind, hidden, dropout):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(d, 1)
        else:
            self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @property
    def out_layer(self):
        return self.net if isinstance(self.net, nn.Linear) else self.net[-1]


class PRISMModel(nn.Module):
    def __init__(self, model_name, genes=GENES, head_type="linear", head_hidden=128, dropout=0.1,
                 chunk_pooling="mean", grad_ckpt=False, encoder=None):
        super().__init__()
        self.genes = list(genes)
        self.encoder = encoder if encoder is not None else load_encoder(model_name)
        d = self.encoder.config.hidden_size
        self.pooler = ChunkPooler(chunk_pooling, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({g: GeneHead(d, head_type, head_hidden, dropout) for g in self.genes})
        if grad_ckpt:
            try:   # non-reentrant: required when lower layers are frozen
                self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception as e:
                print(f"[warn] gradient checkpointing unavailable: {e}")

    def init_biases(self, prevalence):
        """b_g = log(pi / (1 - pi)) from KNOWN training labels."""
        with torch.no_grad():
            for g, pi in zip(self.genes, np.clip(prevalence, 1e-4, 1 - 1e-4)):
                self.heads[g].out_layer.bias.fill_(float(np.log(pi / (1 - pi))))

    def encode_chunks(self, input_ids, attention_mask):
        """(N, L) windows -> (N, d): mean of token vectors inside each window (padding excluded)."""
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1).clamp(min=1)).float()

    def forward(self, input_ids, attention_mask, owner, n_patients):
        v = self.encode_chunks(input_ids, attention_mask)
        z = self.drop(self.norm(self.pooler(v, owner, n_patients)))
        return torch.stack([self.heads[g](z) for g in self.genes], dim=1)        # (B, G) logits


def _layer_list(encoder):
    if hasattr(encoder, "layers"):
        return encoder.layers
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer
    raise AttributeError(f"Cannot find transformer layers on {type(encoder).__name__}")


def set_trainable(encoder, n_top_layers):
    """Freeze the encoder, then unfreeze the top n layers (+ final norm). 0 = fully frozen."""
    for p in encoder.parameters():
        p.requires_grad = False
    if n_top_layers > 0:
        for layer in list(_layer_list(encoder))[-n_top_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        fn = getattr(encoder, "final_norm", None)
        if fn is not None:
            for p in fn.parameters():
                p.requires_grad = True
