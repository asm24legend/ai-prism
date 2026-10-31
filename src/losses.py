import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """ASL with probability shifting (Ridnik et al.). Summed over genes, averaged over batch."""
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, margin=0.05, eps=1e-8):
        super().__init__()
        self.gp, self.gn, self.m, self.eps = gamma_pos, gamma_neg, margin, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits.float())
        p_m = (p - self.m).clamp(min=0)
        lp = y * (1 - p).pow(self.gp) * torch.log(p.clamp(min=self.eps))
        ln = (1 - y) * p_m.pow(self.gn) * torch.log((1 - p_m).clamp(min=self.eps))
        return -(lp + ln).sum(1).mean()


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.register_buffer("pw", pos_weight)

    def forward(self, logits, y):
        return F.binary_cross_entropy_with_logits(
            logits.float(), y, pos_weight=self.pw, reduction="none").sum(1).mean()


def build_loss(cfg, train_prevalence, device):
    if cfg.loss == "asl":
        return AsymmetricLoss(cfg.gamma_pos, cfg.gamma_neg, cfg.margin).to(device)
    pw = None
    if cfg.pos_weight:
        import numpy as np
        prev = np.clip(train_prevalence, 1e-6, None)
        pw = torch.tensor((1 - prev) / prev, dtype=torch.float32, device=device)
    return WeightedBCE(pw).to(device)
