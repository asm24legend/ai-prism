import torch
import torch.nn.functional as F


def masked_bce(logits, y, pos_weight=None):
    """Binary cross-entropy over KNOWN labels only (the specification's L_B). Unknown (NaN) entries contribute
    nothing and receive no gradient. Mean over known entries."""
    mask = ~torch.isnan(y)
    loss = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(y), pos_weight=pos_weight,
                                              reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def make_pos_weight(Y_train, device):
    """(1 - pi) / pi per gene from KNOWN training labels."""
    import numpy as np
    prev = np.clip(np.nanmean(Y_train, axis=0), 1e-6, 1 - 1e-6)
    return torch.tensor((1 - prev) / prev, dtype=torch.float32, device=device)


def no_information_floor(Y_train, pos_weight=None):
    """Loss of a model that predicts each gene's training prevalence for everyone. Real learning shows as the
    training loss falling BELOW this floor (together with a rising training AUROC)."""
    import numpy as np
    prev = np.clip(np.nanmean(Y_train, axis=0), 1e-6, 1 - 1e-6)
    y = torch.tensor(Y_train, dtype=torch.float32)
    logit = torch.tensor(np.log(prev / (1 - prev)), dtype=torch.float32).expand_as(y)
    pw = None if pos_weight is None else pos_weight.detach().cpu()
    return float(masked_bce(logit, y, pw))
