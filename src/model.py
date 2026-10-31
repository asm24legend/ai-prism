"""Clinical ModernBERT encoder + one separate MLP per gene."""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from .data import GENES


class GeneMLP(nn.Module):
    """Gene-specific network: z (768) -> LayerNorm -> H -> H/2 -> 1 logit."""
    def __init__(self, d_in, d_hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PRISMModel(nn.Module):
    def __init__(self, model_name, genes=GENES, head_hidden=256, dropout=0.2,
                 pooling="mean", grad_ckpt=False, encoder=None):
        super().__init__()
        self.genes, self.pooling = list(genes), pooling
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(model_name)
        d = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({g: GeneMLP(d, head_hidden, dropout) for g in self.genes})
        if grad_ckpt:
            try:
                # non-reentrant checkpointing is required when the lower layers are frozen,
                # otherwise the trainable top layers can silently receive no gradients
                self.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception as e:
                print(f"[warn] gradient checkpointing unavailable: {e}")

    def init_biases(self, prevalence):
        """Start each gene's output at its training prevalence: b_g = log(pi / (1 - pi))."""
        with torch.no_grad():
            for g, pi in zip(self.genes, np.clip(prevalence, 1e-4, 1 - 1e-4)):
                self.heads[g].net[-1].bias.fill_(float(np.log(pi / (1 - pi))))

    def embed(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            return h[:, 0]
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask):
        z = self.drop(self.embed(input_ids, attention_mask)).float()
        return torch.stack([self.heads[g](z) for g in self.genes], dim=1)   # (B, G) logits


def _layer_list(encoder):
    """Works for both ModernBertModel (.layers) and BertModel (.encoder.layer)."""
    if hasattr(encoder, "layers"):
        return encoder.layers
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer
    raise AttributeError(f"Cannot find transformer layers on {type(encoder).__name__}")


def set_trainable(encoder, n_top_layers):
    """Freeze the whole encoder, then unfreeze the top n layers (+ final norm if present).
    n_top_layers = 0 keeps the encoder fully frozen (phase 1)."""
    for p in encoder.parameters():
        p.requires_grad = False
    if n_top_layers > 0:
        layers = _layer_list(encoder)
        for layer in list(layers)[-n_top_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        final_norm = getattr(encoder, "final_norm", None)
        if final_norm is not None:
            for p in final_norm.parameters():
                p.requires_grad = True
